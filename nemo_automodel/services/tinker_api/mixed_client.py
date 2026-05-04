# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import pathlib
import re
import uuid
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.services.tinker_api.client import _build_batch
from nemo_automodel.services.tinker_api.future import APIFuture
from nemo_automodel.services.tinker_api.types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    OptimStepResponse,
    SampleResponse,
    SamplingParams,
    SaveStateResponse,
)


def _wildcard_to_regex(pattern: str) -> re.Pattern:
    return re.compile("^" + re.escape(pattern).replace("\\*", ".*") + "$")


def _matches_target(name: str, target_modules: list[str]) -> bool:
    targets = target_modules or ["*_proj"]
    short_name = name.rsplit(".", 1)[-1]
    for pattern in targets:
        if short_name == pattern or name == pattern:
            return True
        if _wildcard_to_regex(pattern).match(name) or _wildcard_to_regex(pattern).match(short_name):
            return True
    return False


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class MixedAdapterLinearLoRA(nn.Module):
    """Linear layer with multiple resident LoRA adapters selected by batch ranges."""

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        lora_dtype: torch.dtype,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = dropout
        self.lora_dtype = lora_dtype
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.active_ranges: list[tuple[str, int, int]] = []

        for param in self.base_linear.parameters():
            param.requires_grad_(False)

    @property
    def in_features(self) -> int:
        """Input feature size."""
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        """Output feature size."""
        return self.base_linear.out_features

    def add_adapter(self, adapter_id: str) -> None:
        """Add one resident LoRA adapter to this linear layer."""
        if adapter_id in self.lora_a:
            return
        lora_a = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=self.base_linear.weight.device, dtype=self.lora_dtype)
        )
        lora_b = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=self.base_linear.weight.device, dtype=self.lora_dtype)
        )
        nn.init.kaiming_uniform_(lora_a, a=5**0.5)
        self.lora_a[adapter_id] = lora_a
        self.lora_b[adapter_id] = lora_b

    def set_active_ranges(self, ranges: list[tuple[str, int, int]]) -> None:
        """Set batch row ranges for the next mixed-adapter forward pass."""
        self.active_ranges = ranges

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        """Return trainable parameters for one adapter."""
        return [self.lora_a[adapter_id], self.lora_b[adapter_id]]

    def adapter_state_dict(self, adapter_id: str, prefix: str) -> dict[str, torch.Tensor]:
        """Return one adapter's state for this layer."""
        return {
            f"{prefix}.lora_a": self.lora_a[adapter_id].detach().cpu(),
            f"{prefix}.lora_b": self.lora_b[adapter_id].detach().cpu(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base linear layer and the selected LoRA adapters."""
        result = self.base_linear(x)
        if not self.active_ranges:
            return result

        for adapter_id, start_idx, end_idx in self.active_ranges:
            if adapter_id not in self.lora_a:
                continue
            x_slice = x[start_idx:end_idx]
            if self.training and self.dropout > 0:
                x_slice = F.dropout(x_slice, p=self.dropout, training=True)
            lora_out = F.linear(x_slice.to(self.lora_dtype), self.lora_a[adapter_id])
            lora_out = F.linear(lora_out, self.lora_b[adapter_id]) * self.scale
            result[start_idx:end_idx] = result[start_idx:end_idx] + lora_out.to(result.dtype)
        return result


@dataclass
class MixedAdapterHandle:
    """One resident adapter plus its optimizer state."""

    adapter_id: str
    optimizer: Optional[torch.optim.AdamW] = None
    step: int = 0


class MixedLoraServiceClient:
    """Single-node mLoRA-style service with multiple resident adapters."""

    def __init__(
        self,
        *,
        base_model: str,
        scratch_dir: str | pathlib.Path = "/home/scratch.asteiner",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str | torch.dtype = "bfloat16",
        trust_remote_code: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.scratch_dir = pathlib.Path(scratch_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lora_config = lora_config or LoraConfig(rank=16)
        self.adapters: dict[str, MixedAdapterHandle] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        self.model.to(self.device)
        self.model.train()
        self.mixed_lora_layers = self._patch_target_linears()

    def _patch_target_linears(self) -> dict[str, MixedAdapterLinearLoRA]:
        layers = {}
        rank = self.lora_config.rank
        alpha = self.lora_config.alpha or rank * 2
        lora_dtype = next(self.model.parameters()).dtype
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not _matches_target(name, self.lora_config.target_modules):
                continue
            if self.lora_config.exclude_modules and _matches_target(name, self.lora_config.exclude_modules):
                continue
            parent, child_name = _get_parent_module(self.model, name)
            mixed_layer = MixedAdapterLinearLoRA(
                module,
                rank=rank,
                alpha=alpha,
                dropout=self.lora_config.dropout,
                lora_dtype=lora_dtype,
            )
            setattr(parent, child_name, mixed_layer)
            layers[name] = mixed_layer
        if not layers:
            raise ValueError("No target nn.Linear modules were found for mixed LoRA")
        return layers

    def create_lora_training_client(self) -> "MixedLoraTrainingClient":
        """Create one resident LoRA adapter training client."""
        adapter_id = f"adapter_{uuid.uuid4().hex[:12]}"
        for layer in self.mixed_lora_layers.values():
            layer.add_adapter(adapter_id)
        handle = MixedAdapterHandle(adapter_id=adapter_id)
        self.adapters[adapter_id] = handle
        return MixedLoraTrainingClient(service=self, handle=handle)

    def _set_active_ranges(self, ranges: list[tuple[str, int, int]]) -> None:
        for layer in self.mixed_lora_layers.values():
            layer.set_active_ranges(ranges)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        """Return trainable parameters for one adapter."""
        params = []
        for layer in self.mixed_lora_layers.values():
            params.extend(layer.adapter_parameters(adapter_id))
        return params

    def adapter_state_dict(self, adapter_id: str) -> dict[str, torch.Tensor]:
        """Return one adapter's resident weights."""
        state = {}
        for name, layer in self.mixed_lora_layers.items():
            state.update(layer.adapter_state_dict(adapter_id, name))
        return state

    def forward_backward_mixed(
        self,
        batches_by_adapter: dict[str, list[Datum]],
        loss_fn: str = "cross_entropy",
    ) -> APIFuture[dict[str, ForwardBackwardOutput]]:
        """Run one mixed-adapter forward/backward pass over a concatenated batch."""
        if loss_fn != "cross_entropy":
            raise NotImplementedError("Mixed LoRA prototype only supports loss_fn='cross_entropy'")
        if not batches_by_adapter:
            raise ValueError("forward_backward_mixed requires at least one adapter batch")

        data = []
        ranges = []
        start_idx = 0
        adapter_order = []
        for adapter_id, batch in batches_by_adapter.items():
            if adapter_id not in self.adapters:
                raise KeyError(f"Unknown adapter_id: {adapter_id}")
            if not batch:
                continue
            data.extend(batch)
            end_idx = start_idx + len(batch)
            ranges.append((adapter_id, start_idx, end_idx))
            adapter_order.append(adapter_id)
            start_idx = end_idx
        if not data:
            raise ValueError("At least one adapter batch must be non-empty")

        input_ids, labels = _build_batch(data, self.tokenizer.pad_token_id, self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(torch.long)

        self.model.zero_grad(set_to_none=True)
        self._set_active_ranges(ranges)
        self.model.train()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        token_mask = shift_labels.ne(-100)

        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        outputs_by_adapter = {}
        with torch.no_grad():
            target_logprobs = F.log_softmax(shift_logits, dim=-1)
            safe_labels = shift_labels.clamp_min(0).unsqueeze(-1)
            gathered = target_logprobs.gather(-1, safe_labels).squeeze(-1)
            gathered = gathered.masked_fill(~token_mask, 0.0)

        for adapter_id, start, end in ranges:
            adapter_loss = per_token_loss[start:end]
            adapter_mask = token_mask[start:end]
            denom = adapter_mask.sum().clamp_min(1)
            mean_loss = adapter_loss.sum() / denom
            total_loss = total_loss + mean_loss
            outputs_by_adapter[adapter_id] = ForwardBackwardOutput(
                loss=float(mean_loss.detach().cpu()),
                metrics={"loss": float(mean_loss.detach().cpu()), "num_label_tokens": float(denom.detach().cpu())},
                loss_fn_outputs=[{"logprobs": row.detach().cpu().tolist()} for row in gathered[start:end]],
            )

        total_loss.backward()
        self._set_active_ranges([])
        return APIFuture(outputs_by_adapter)

    def optim_step(self, adapter_id: str, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Apply gradients for one resident adapter."""
        handle = self.adapters[adapter_id]
        if handle.optimizer is None:
            handle.optimizer = torch.optim.AdamW(
                self.adapter_parameters(adapter_id),
                lr=adam_params.learning_rate,
                betas=adam_params.betas,
                eps=adam_params.eps,
                weight_decay=adam_params.weight_decay,
            )
        else:
            for group in handle.optimizer.param_groups:
                group["lr"] = adam_params.learning_rate
                group["weight_decay"] = adam_params.weight_decay
                group["betas"] = adam_params.betas
                group["eps"] = adam_params.eps

        handle.optimizer.step()
        handle.optimizer.zero_grad(set_to_none=True)
        handle.step += 1
        return APIFuture(OptimStepResponse(step=handle.step, learning_rate=adam_params.learning_rate))

    def save_adapter_state(self, adapter_id: str, name: str) -> APIFuture[SaveStateResponse]:
        """Save one resident adapter's weights and optimizer state."""
        output_dir = self.scratch_dir / "checkpoints" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state_dict(adapter_id), output_dir / "adapter_model.pt")
        handle = self.adapters[adapter_id]
        if handle.optimizer is not None:
            torch.save(handle.optimizer.state_dict(), output_dir / "optimizer.pt")
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "base_model": self.base_model,
                    "adapter_id": adapter_id,
                    "rank": self.lora_config.rank,
                    "alpha": self.lora_config.alpha,
                    "target_modules": self.lora_config.target_modules or ["*_proj"],
                    "step": handle.step,
                },
                fp,
                indent=2,
            )
        return APIFuture(SaveStateResponse(path=str(output_dir)))

    def sample(
        self, adapter_id: str, prompt: str, params: Optional[SamplingParams] = None
    ) -> APIFuture[SampleResponse]:
        """Generate with one resident adapter selected for the full prompt batch."""
        params = params or SamplingParams()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._set_active_ranges([(adapter_id, 0, 1)])
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                top_p=params.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
        self._set_active_ranges([])
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return APIFuture(SampleResponse(tokens=output_ids.detach().cpu().tolist(), text=text))


class MixedLoraTrainingClient:
    """One resident adapter client for `MixedLoraServiceClient`."""

    def __init__(self, *, service: MixedLoraServiceClient, handle: MixedAdapterHandle):
        self.service = service
        self.handle = handle
        self.adapter_id = handle.adapter_id

    def optim_step(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Apply this adapter's gradients."""
        return self.service.optim_step(self.adapter_id, adam_params)

    def save_state(self, name: str) -> APIFuture[SaveStateResponse]:
        """Save this adapter's state."""
        return self.service.save_adapter_state(self.adapter_id, name)
