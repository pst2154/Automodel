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

import pytest
import torch

from nemo_automodel.services.tinker_api import client as tinker_client
from nemo_automodel.services.tinker_api.client import _build_batch
from nemo_automodel.services.tinker_api.future import APIFuture
from nemo_automodel.services.tinker_api.mixed_client import (
    MixedAdapterLinearLoRA,
    MixedLoraServiceClient,
    _rl_token_loss,
)
from nemo_automodel.services.tinker_api.types import Datum, ModelInput


def test_api_future_returns_value():
    assert APIFuture(7).result() == 7


def test_build_batch_pads_and_masks_weights():
    data = [
        Datum(
            model_input=ModelInput.from_ints([10, 11, 12]),
            loss_fn_inputs={"target_tokens": ModelInput.from_ints([10, 11, 12]), "weights": [0, 1, 1]},
        ),
        Datum(model_input=ModelInput.from_ints([20, 21]), loss_fn_inputs={}),
    ]

    input_ids, labels = _build_batch(data, pad_token_id=0, device=torch.device("cpu"))

    assert input_ids.tolist() == [[10, 11, 12], [20, 21, 0]]
    assert labels.tolist() == [[-100, 11, 12], [20, 21, -100]]


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo", "dro"])
def test_rl_token_loss_modes_return_finite_weighted_loss(loss_fn):
    data = [
        Datum(
            model_input=ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "weights": [1.0, 1.0],
                "advantages": [1.5, -0.5],
                "logprobs": [-1.1, -1.2],
            },
        ),
        Datum(
            model_input=ModelInput.from_ints([4, 5, 6]),
            loss_fn_inputs={
                "weights": [1.0, 0.0],
                "advantages": [0.25, 0.0],
                "logprobs": [-0.9, 0.0],
            },
        ),
    ]
    per_token_loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gathered = torch.tensor([[-1.0, -1.4], [-0.7, -0.1]])
    token_mask = torch.tensor([[True, True], [True, False]])

    token_loss, effective_mask, metrics = _rl_token_loss(
        loss_fn=loss_fn,
        per_token_loss=per_token_loss,
        gathered_logprobs=gathered,
        data=data,
        token_mask=token_mask,
        device=torch.device("cpu"),
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2, "beta": 0.05},
    )
    ratio = torch.exp(gathered - torch.tensor([[-1.1, -1.2], [-0.9, 0.0]]))
    advantages = torch.tensor([[1.5, -0.5], [0.25, 0.0]])
    weights = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    if loss_fn == "importance_sampling":
        expected = -ratio * advantages * weights
    elif loss_fn == "ppo":
        expected = -torch.minimum(ratio * advantages * weights, ratio.clamp(0.8, 1.2) * advantages * weights)
    elif loss_fn == "cispo":
        expected = -(ratio.clamp(0.8, 1.2) * gathered * advantages * weights)
    else:
        expected = (
            -((gathered * advantages) - 0.5 * 0.05 * (gathered - torch.tensor([[-1.1, -1.2], [-0.9, 0.0]])).pow(2))
            * weights
        )

    assert token_loss.shape == per_token_loss.shape
    assert torch.isfinite(token_loss[effective_mask]).all()
    assert effective_mask.tolist() == [[True, True], [True, False]]
    assert metrics["loss_weight_mean"] == 1.0
    assert torch.allclose(token_loss, expected)


def test_mixed_lora_layer_routes_ranges_with_torch_fallback():
    base = torch.nn.Linear(3, 2, bias=False)
    base.weight.data.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    layer = MixedAdapterLinearLoRA(base, rank=1, alpha=2, dropout=0.0, lora_dtype=torch.float32, use_triton_lora=True)
    layer.add_adapter("atlas")
    layer.add_adapter("borealis")
    layer.lora_a["atlas"].data.copy_(torch.tensor([[1.0, 1.0, 0.0]]))
    layer.lora_b["atlas"].data.copy_(torch.tensor([[1.0], [0.0]]))
    layer.lora_a["borealis"].data.copy_(torch.tensor([[0.0, 1.0, 1.0]]))
    layer.lora_b["borealis"].data.copy_(torch.tensor([[0.0], [1.0]]))
    layer.set_active_ranges([("atlas", 0, 1), ("borealis", 1, 2)])

    out = layer(torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]))

    assert not layer._can_use_triton_lora(torch.zeros(1, 3))
    assert torch.allclose(out, torch.tensor([[12.0, 3.0], [7.0, 59.0]]))


def test_mixed_lora_layer_grouped_backend_matches_loop_forward_backward():
    torch.manual_seed(1234)
    base_loop = torch.nn.Linear(8, 6, bias=False)
    base_grouped = torch.nn.Linear(8, 6, bias=False)
    base_grouped.weight.data.copy_(base_loop.weight)

    layer_loop = MixedAdapterLinearLoRA(
        base_loop,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="loop",
    )
    layer_grouped = MixedAdapterLinearLoRA(
        base_grouped,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="grouped",
    )
    for adapter_id in ["atlas", "borealis"]:
        layer_loop.add_adapter(adapter_id)
        layer_grouped.add_adapter(adapter_id)
        layer_loop.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_loop.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_grouped.lora_a[adapter_id].data.copy_(layer_loop.lora_a[adapter_id])
        layer_grouped.lora_b[adapter_id].data.copy_(layer_loop.lora_b[adapter_id])

    ranges = [("atlas", 0, 2), ("borealis", 2, 4)]
    layer_loop.set_active_ranges(ranges)
    layer_grouped.set_active_ranges(ranges)
    x_loop = torch.randn(4, 5, 8, requires_grad=True)
    x_grouped = x_loop.detach().clone().requires_grad_(True)

    out_loop = layer_loop(x_loop)
    out_grouped = layer_grouped(x_grouped)
    out_loop.pow(2).sum().backward()
    out_grouped.pow(2).sum().backward()

    assert layer_grouped._can_use_grouped_lora(x_grouped)
    assert torch.allclose(out_grouped, out_loop)
    assert torch.allclose(x_grouped.grad, x_loop.grad)
    for adapter_id in ["atlas", "borealis"]:
        assert torch.allclose(layer_grouped.lora_a[adapter_id].grad, layer_loop.lora_a[adapter_id].grad)
        assert torch.allclose(layer_grouped.lora_b[adapter_id].grad, layer_loop.lora_b[adapter_id].grad)


@pytest.mark.run_only_on("GPU")
def test_mixed_lora_layer_triton_bridge_matches_torch_forward_backward():
    torch.manual_seed(1234)
    base_ref = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_triton = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_triton.weight.data.copy_(base_ref.weight)

    layer_ref = MixedAdapterLinearLoRA(
        base_ref,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        use_triton_lora=False,
    )
    layer_triton = MixedAdapterLinearLoRA(
        base_triton,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        use_triton_lora=True,
    )
    for adapter_id in ["atlas", "borealis"]:
        layer_ref.add_adapter(adapter_id)
        layer_triton.add_adapter(adapter_id)
        layer_ref.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_ref.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_triton.lora_a[adapter_id].data.copy_(layer_ref.lora_a[adapter_id])
        layer_triton.lora_b[adapter_id].data.copy_(layer_ref.lora_b[adapter_id])

    ranges = [("atlas", 0, 1), ("borealis", 1, 3)]
    layer_ref.set_active_ranges(ranges)
    layer_triton.set_active_ranges(ranges)
    x_ref = torch.randn(3, 4, 8, device="cuda", requires_grad=True)
    x_triton = x_ref.detach().clone().requires_grad_(True)

    out_ref = layer_ref(x_ref)
    out_triton = layer_triton(x_triton)
    out_ref.pow(2).sum().backward()
    out_triton.pow(2).sum().backward()

    assert layer_triton._can_use_triton_lora(x_triton)
    assert torch.allclose(out_triton, out_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(x_triton.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    for adapter_id in ["atlas", "borealis"]:
        assert torch.allclose(
            layer_triton.lora_a[adapter_id].grad,
            layer_ref.lora_a[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            layer_triton.lora_b[adapter_id].grad,
            layer_ref.lora_b[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.run_only_on("GPU")
def test_mixed_lora_layer_grouped_triton_matches_torch_forward_backward():
    torch.manual_seed(1234)
    base_ref = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_grouped_triton = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_grouped_triton.weight.data.copy_(base_ref.weight)

    layer_ref = MixedAdapterLinearLoRA(
        base_ref,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="loop",
    )
    layer_grouped_triton = MixedAdapterLinearLoRA(
        base_grouped_triton,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="grouped_triton",
    )
    for adapter_id in ["atlas", "borealis", "cygnus"]:
        layer_ref.add_adapter(adapter_id)
        layer_grouped_triton.add_adapter(adapter_id)
        layer_ref.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_ref.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_grouped_triton.lora_a[adapter_id].data.copy_(layer_ref.lora_a[adapter_id])
        layer_grouped_triton.lora_b[adapter_id].data.copy_(layer_ref.lora_b[adapter_id])

    ranges = [("atlas", 0, 1), ("borealis", 1, 3), ("cygnus", 3, 4)]
    layer_ref.set_active_ranges(ranges)
    layer_grouped_triton.set_active_ranges(ranges)
    x_ref = torch.randn(4, 5, 8, device="cuda", requires_grad=True)
    x_grouped_triton = x_ref.detach().clone().requires_grad_(True)

    out_ref = layer_ref(x_ref)
    out_grouped_triton = layer_grouped_triton(x_grouped_triton)
    out_ref.pow(2).sum().backward()
    out_grouped_triton.pow(2).sum().backward()

    assert layer_grouped_triton._can_use_grouped_triton_lora(x_grouped_triton)
    assert torch.allclose(out_grouped_triton, out_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(x_grouped_triton.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    for adapter_id in ["atlas", "borealis", "cygnus"]:
        assert torch.allclose(
            layer_grouped_triton.lora_a[adapter_id].grad,
            layer_ref.lora_a[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            layer_grouped_triton.lora_b[adapter_id].grad,
            layer_ref.lora_b[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )


def test_service_reuses_worker_for_matching_base_and_lora_config(monkeypatch, tmp_path):
    created_workers = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.base_model = kwargs["base_model"]
            self.lora_config = kwargs["lora_config"]
            self.device = torch.device("cpu")
            created_workers.append(self)

        def new_adapter_state(self):
            return {}

    monkeypatch.setattr(tinker_client, "SharedBaseModelWorker", FakeWorker)

    service = tinker_client.ServiceClient(scratch_dir=tmp_path)
    first = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)
    second = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)

    assert first.worker is second.worker
    assert len(created_workers) == 1


def test_service_separates_workers_for_different_lora_config(monkeypatch, tmp_path):
    created_workers = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.base_model = kwargs["base_model"]
            self.lora_config = kwargs["lora_config"]
            self.device = torch.device("cpu")
            created_workers.append(self)

        def new_adapter_state(self):
            return {}

    monkeypatch.setattr(tinker_client, "SharedBaseModelWorker", FakeWorker)

    service = tinker_client.ServiceClient(scratch_dir=tmp_path)
    first = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)
    second = service.create_lora_training_client("tiny-model", rank=8, force_hf=True)

    assert first.worker is not second.worker
    assert len(created_workers) == 2


def test_mixed_lora_checkpoint_validation_rejects_wrong_base_model(tmp_path):
    service = MixedLoraServiceClient.__new__(MixedLoraServiceClient)
    service.base_model = "expected-model"
    service.lora_config = type(
        "FakeLoraConfig",
        (),
        {"rank": 16, "alpha": None, "target_modules": []},
    )()

    try:
        service._validate_checkpoint_config(
            {
                "base_model": "other-model",
                "rank": 16,
                "alpha": None,
                "target_modules": ["*_proj"],
            },
            tmp_path,
        )
    except ValueError as exc:
        assert "base_model" in str(exc)
    else:
        raise AssertionError("Expected checkpoint validation to reject mismatched base model")
