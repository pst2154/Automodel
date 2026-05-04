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

import torch

from nemo_automodel.services.tinker_api import client as tinker_client
from nemo_automodel.services.tinker_api.client import _build_batch
from nemo_automodel.services.tinker_api.future import APIFuture
from nemo_automodel.services.tinker_api.mixed_client import MixedLoraServiceClient
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
