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

from nemo_automodel.services.tinker_api import server
from nemo_automodel.services.tinker_api.future import APIFuture
from nemo_automodel.services.tinker_api.types import (
    ForwardBackwardOutput,
    OptimStepResponse,
    SampleResponse,
    SaveStateResponse,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")


class FakeTrainingClient:
    def __init__(self, service, adapter_id):
        self.service = service
        self.adapter_id = adapter_id

    def optim_step(self, adam_params):
        self.service.steps[self.adapter_id] += 1
        return APIFuture(
            OptimStepResponse(step=self.service.steps[self.adapter_id], learning_rate=adam_params.learning_rate)
        )

    def save_state(self, name):
        return APIFuture(SaveStateResponse(path=f"/tmp/{name}"))


class FakeMixedLoraServiceClient:
    def __init__(self, **kwargs):
        self.created = 0
        self.steps = {}

    def create_lora_training_client(self):
        self.created += 1
        adapter_id = f"adapter_{self.created}"
        self.steps[adapter_id] = 0
        return FakeTrainingClient(self, adapter_id)

    def forward_backward_mixed(self, batches_by_adapter, loss_fn):
        outputs = {}
        for adapter_id, batch in batches_by_adapter.items():
            outputs[adapter_id] = ForwardBackwardOutput(
                loss=float(len(batch)),
                metrics={"loss": float(len(batch)), "num_label_tokens": 3.0},
                loss_fn_outputs=[{"logprobs": [-1.0, -0.5]}],
            )
        return APIFuture(outputs)

    def sample(self, adapter_id, prompt, params):
        return APIFuture(SampleResponse(tokens=[1, 2, 3], text=f"{prompt} {adapter_id}"))


def test_mixed_lora_server_tracks_run_lifecycle(monkeypatch):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model")
    client = fastapi_testclient.TestClient(app)

    first = client.post("/runs", json={"name": "atlas"}).json()
    second = client.post("/runs", json={"name": "borealis"}).json()

    response = client.post(
        "/mixed_forward_backward",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            }
        },
    ).json()

    assert response[first["run_id"]]["run"]["status"] == "ready"
    assert response[first["run_id"]]["run"]["forward_backward_calls"] == 1
    assert response[first["run_id"]]["run"]["last_loss"] == 1.0

    step = client.post(f"/runs/{first['run_id']}/optim_step", json={"learning_rate": 0.001}).json()
    assert step["run"]["optimizer_steps"] == 1
    assert step["output"]["learning_rate"] == 0.001

    sample = client.post(f"/runs/{first['run_id']}/sample", json={"prompt": "hello"}).json()
    assert sample["output"]["text"] == "hello adapter_1"

    saved = client.post(f"/runs/{first['run_id']}/save", json={"name": "atlas-test"}).json()
    assert saved["run"]["last_checkpoint_path"] == "/tmp/atlas-test"

    record = client.get(f"/runs/{first['run_id']}").json()
    assert record["sequence"] >= 4
    assert record["last_checkpoint_path"] == "/tmp/atlas-test"
