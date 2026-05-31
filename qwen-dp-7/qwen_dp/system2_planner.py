"""Frozen Qwen2-VL System-2 encoder with a learnable subgoal token."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass(frozen=True)
class System2PlannerConfig:
    model_path: str = "models/Qwen2-VL-7B"
    subgoal_token: str = "<SUBGOAL>"
    torch_dtype: str = "bfloat16"
    device_map: str | dict[str, int | str] | None = "auto"
    trust_remote_code: bool = True
    use_fast_processor: bool = True
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen linear projection."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(p=dropout)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        device = self.base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        base_output = self.base(input)
        lora_input = self.dropout(input).to(dtype=self.lora_A.dtype)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        return base_output + lora_output.to(dtype=base_output.dtype) * self.scaling


class System2Planner(nn.Module):
    """Step-5 System 2: Qwen2-VL frozen except the new subgoal token.

    The Qwen token embedding matrix is one large parameter, so PyTorch cannot
    mark only one row as trainable with `requires_grad`. We keep that parameter
    trainable and install a gradient hook that zeros every row except the
    `<SUBGOAL>` token row.
    """

    def __init__(self, config: System2PlannerConfig = System2PlannerConfig()):
        super().__init__()
        self.config = config
        self.model_path = Path(config.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=config.use_fast_processor,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=_resolve_dtype(config.torch_dtype),
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )

        self.subgoal_token_id = self._register_subgoal_token(config.subgoal_token)
        self._freeze_base_model_except_subgoal_token()
        if config.lora_r > 0:
            self._install_lora_adapters()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _register_subgoal_token(self, token: str) -> int:
        tokenizer = self.processor.tokenizer
        added = tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        if added:
            self.model.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Could not register subgoal token: {token}")
        return int(token_id)

    def _freeze_base_model_except_subgoal_token(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        embedding = self.model.get_input_embeddings()
        embedding.weight.requires_grad = True

        token_id = self.subgoal_token_id

        def keep_only_subgoal_row(gradient: Tensor) -> Tensor:
            masked = torch.zeros_like(gradient)
            masked[token_id] = gradient[token_id]
            return masked

        embedding.weight.register_hook(keep_only_subgoal_row)

    def _install_lora_adapters(self) -> None:
        replaced = 0
        targets = tuple(self.config.lora_target_modules)
        for module_name, module in list(self.model.named_modules()):
            if not module_name.endswith(targets) or not isinstance(module, nn.Linear):
                continue

            parent_name, child_name = module_name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
            setattr(
                parent,
                child_name,
                LoRALinear(
                    module,
                    r=self.config.lora_r,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                ),
            )
            replaced += 1

        if replaced == 0:
            raise ValueError(f"No Linear modules matched LoRA targets: {targets}")

    def lora_state_dict(self) -> dict[str, Tensor]:
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if ".lora_" in name
        }

    def load_lora_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        if not state_dict:
            return
        current = dict(self.named_parameters())
        missing = sorted(name for name in state_dict if name not in current)
        if missing:
            raise ValueError(
                "Checkpoint contains LoRA weights, but this planner does not have matching adapters. "
                "Pass the same --system2-lora-* args used during Stage 2. "
                f"First missing key: {missing[0]}"
            )

        with torch.no_grad():
            for name, tensor in state_dict.items():
                parameter = current[name]
                if parameter.shape != tensor.shape:
                    raise ValueError(
                        f"LoRA shape mismatch for {name}: expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                    )
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.named_parameters() if param.requires_grad]

    def subgoal_embedding(self) -> Tensor:
        embedding = self.model.get_input_embeddings()
        return embedding.weight[self.subgoal_token_id]

    def forward(self, **model_inputs: Tensor) -> Tensor:
        """Return the final hidden state at the `<SUBGOAL>` token position.

        `model_inputs` should come from `self.processor(...)` and must include
        `input_ids` containing the `<SUBGOAL>` token.
        """
        if "input_ids" not in model_inputs:
            raise ValueError("model_inputs must include input_ids.")

        input_ids = model_inputs["input_ids"]
        subgoal_mask = input_ids == self.subgoal_token_id
        if not bool(subgoal_mask.any()):
            raise ValueError(f"input_ids do not contain {self.config.subgoal_token}.")

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]

        subgoal_mask = subgoal_mask.to(hidden.device)
        subgoal_vectors = []
        for batch_idx in range(input_ids.shape[0]):
            positions = torch.nonzero(subgoal_mask[batch_idx], as_tuple=False).flatten()
            if len(positions) == 0:
                raise ValueError(f"Batch item {batch_idx} does not contain {self.config.subgoal_token}.")
            subgoal_vectors.append(hidden[batch_idx, positions[-1]])
        return torch.stack(subgoal_vectors, dim=0)
