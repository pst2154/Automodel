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

import time
from types import SimpleNamespace

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
    def __init__(self, service, adapter_id, step=0):
        self.service = service
        self.adapter_id = adapter_id
        self.handle = SimpleNamespace(step=step)

    def optim_step(self, adam_params):
        self.service.steps[self.adapter_id] += 1
        self.handle.step = self.service.steps[self.adapter_id]
        return APIFuture(
            OptimStepResponse(step=self.service.steps[self.adapter_id], learning_rate=adam_params.learning_rate)
        )

    def save_state(self, name):
        return APIFuture(SaveStateResponse(path=f"/tmp/{name}"))


class FakeMixedLoraServiceClient:
    def __init__(self, **kwargs):
        self.created = 0
        self.steps = {}

    def create_lora_training_client(self, *, adapter_id=None, checkpoint_path=None):
        self.created += 1
        adapter_id = adapter_id or f"adapter_{self.created}"
        self.steps[adapter_id] = 0
        if checkpoint_path is not None:
            self.steps[adapter_id] = 7
        return FakeTrainingClient(self, adapter_id, step=self.steps[adapter_id])

    def forward_backward_mixed(self, batches_by_adapter, loss_fn, loss_fn_config=None):
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


def test_mixed_lora_server_tracks_run_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, use_triton_lora=True)
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()
    assert health["use_triton_lora"] is True
    assert health["mixed_lora_backend"] == "triton"
    assert health["metadata_backend"] == "sqlite"
    assert health["max_runs_per_tenant"] is None
    assert health["tenant_rate_limit_per_minute"] is None
    assert health["restore_runs_on_startup"] is False
    assert health["resume_interrupted_jobs_on_startup"] is False

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


def test_mixed_lora_server_reports_supervised_worker_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=2)
    with fastapi_testclient.TestClient(app) as client:
        health = client.get("/health").json()
        workers = client.get("/workers").json()
        created = client.post("/runs", json={"name": "placed"}).json()
        record = client.get(f"/runs/{created['run_id']}").json()
        ping = client.post(f"/workers/{created['worker_id']}/ping").json()

        assert health["worker_processes"] == 2
        assert len(health["workers"]) == 2
        assert len(workers) == 2
        assert {worker["status"] for worker in workers} == {"running"}
        assert all(worker["pid"] for worker in workers)
        assert created["worker_id"] in {worker["worker_id"] for worker in workers}
        assert record["worker_id"] == created["worker_id"]
        assert ping["worker"]["worker_id"] == created["worker_id"]
        assert ping["result"]["worker_pid"] == ping["worker"]["pid"]


def test_mixed_lora_server_restores_run_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    restored = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_restored", "checkpoint_path": "/tmp/checkpoint"},
    ).json()

    record = client.get(f"/runs/{restored['run_id']}").json()
    assert restored["adapter_id"] == "adapter_restored"
    assert restored["status"] == "ready"
    assert record["optimizer_steps"] == 7
    assert record["restored_from"] == "/tmp/checkpoint"


def test_mixed_lora_server_marks_persisted_runs_detached(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    restarted = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    restarted_client = fastapi_testclient.TestClient(restarted)

    record = restarted_client.get(f"/runs/{created['run_id']}").json()
    assert record["status"] == "detached"


def test_mixed_lora_server_rehydrates_checkpointed_runs_on_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_restored", "checkpoint_path": "/tmp/checkpoint"},
    ).json()

    restarted = server.create_app(base_model="fake-model", scratch_dir=tmp_path, restore_runs_on_startup=True)
    restarted_client = fastapi_testclient.TestClient(restarted)

    health = restarted_client.get("/health").json()
    record = restarted_client.get(f"/runs/{created['run_id']}").json()
    sample = restarted_client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "hello"}).json()

    assert health["restore_runs_on_startup"] is True
    assert record["status"] == "ready"
    assert record["optimizer_steps"] == 7
    assert record["restored_from"] == "/tmp/checkpoint"
    assert sample["output"]["text"] == "hello adapter_restored"


def test_mixed_lora_server_can_use_json_metadata_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, metadata_backend="json")
    client = fastapi_testclient.TestClient(app)

    created = client.post("/runs", json={"name": "json-run"}).json()
    health = client.get("/health").json()

    assert created["status"] == "created"
    assert health["metadata_backend"] == "json"
    assert (tmp_path / "tinker_api" / "runs.json").exists()


def test_mixed_lora_server_runs_server_owned_train_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()
    second = client.post("/runs", json={"name": "borealis"}).json()

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            },
            "steps": 3,
            "learning_rate": 0.001,
            "save_names": {first["run_id"]: "atlas-job"},
        },
    ).json()

    assert response["job"]["status"] == "succeeded"
    assert response["job"]["progress"]["step"] == 3
    assert response["job"]["result"]["last_losses"][first["run_id"]] == 1.0
    assert response["runs"][first["run_id"]]["optimizer_steps"] == 3
    assert response["runs"][first["run_id"]]["last_checkpoint_path"] == "/tmp/atlas-job"

    jobs = client.get("/jobs").json()
    assert jobs[0]["kind"] == "train_steps"


def test_mixed_lora_server_submits_async_train_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()

    submitted = client.post(
        "/train_steps",
        json={
            "batches": {first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
            "steps": 1,
            "learning_rate": 0.001,
            "run_async": True,
        },
    ).json()

    job = client.get(f"/jobs/{submitted['job']['job_id']}").json()
    assert job["status"] in {"queued", "running", "succeeded"}


def test_mixed_lora_server_resumes_interrupted_train_job_on_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_resumed", "checkpoint_path": "/tmp/checkpoint"},
    ).json()
    request_payload = {
        "batches": {created["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
        "steps": 3,
        "learning_rate": 0.001,
    }
    now = server._utc_now()
    job = server.JobRecord(
        job_id="job_resume",
        kind="train_steps",
        status="running",
        run_ids=[created["run_id"]],
        progress={"step": 1, "total_steps": 3, "request": request_payload},
        created_at=now,
        updated_at=now,
    )
    store = server.SQLiteStore(tmp_path / "tinker_api" / "metadata.sqlite3", "jobs", server.JobRecord)
    store.save({"job_resume": job})

    restarted = server.create_app(
        base_model="fake-model",
        scratch_dir=tmp_path,
        restore_runs_on_startup=True,
        resume_interrupted_jobs_on_startup=True,
    )
    restarted_client = fastapi_testclient.TestClient(restarted)

    resumed = restarted_client.get("/jobs/job_resume").json()
    for _ in range(20):
        if resumed["status"] == "succeeded":
            break
        time.sleep(0.05)
        resumed = restarted_client.get("/jobs/job_resume").json()
    run = restarted_client.get(f"/runs/{created['run_id']}").json()

    assert resumed["status"] == "succeeded"
    assert resumed["progress"]["step"] == 3
    assert run["optimizer_steps"] == 9


def test_mixed_lora_server_reuses_idempotent_create_response(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    first = client.post("/runs", json={"name": "atlas", "idempotency_key": "create-atlas"}).json()
    second = client.post("/runs", json={"name": "atlas", "idempotency_key": "create-atlas"}).json()

    assert second == first
    assert len(client.get("/runs").json()) == 1


def test_mixed_lora_server_rejects_idempotency_key_reuse_for_different_request(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas", "idempotency_key": "same-key"}).status_code == 200
    response = client.post("/runs", json={"name": "borealis", "idempotency_key": "same-key"})

    assert response.status_code == 409


def test_mixed_lora_server_reuses_idempotent_train_steps_response(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()

    payload = {
        "batches": {first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
        "steps": 2,
        "learning_rate": 0.001,
        "idempotency_key": "train-atlas",
    }
    first_response = client.post("/train_steps", json=payload).json()
    second_response = client.post("/train_steps", json=payload).json()

    assert second_response == first_response
    assert client.get(f"/runs/{first['run_id']}").json()["optimizer_steps"] == 2


def test_mixed_lora_server_requires_bearer_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, api_key="secret")
    client = fastapi_testclient.TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/runs", json={"name": "atlas"}).status_code == 401
    authorized = client.post("/runs", json={"name": "atlas"}, headers={"Authorization": "Bearer secret"})

    assert authorized.status_code == 200


def test_mixed_lora_server_enforces_resident_adapter_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, max_resident_adapters=1)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas"}).status_code == 200
    response = client.post("/runs", json={"name": "borealis"})

    assert response.status_code == 429


def test_mixed_lora_server_enforces_tenant_adapter_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, max_runs_per_tenant=1)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).status_code == 200
    assert client.post("/runs", json={"name": "borealis", "tenant_id": "tenant-b"}).status_code == 200
    response = client.post("/runs", json={"name": "orion", "tenant_id": "tenant-a"})

    assert response.status_code == 429
    assert "tenant-a" in response.json()["detail"]


def test_mixed_lora_server_rate_limits_tenant_gpu_operations(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, tenant_rate_limit_per_minute=2)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).json()

    first = client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "hello"})
    second = client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "again"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "tenant-a" in second.json()["detail"]


def test_mixed_lora_server_records_tenant_and_rejects_mixed_tenant_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).json()
    second = client.post("/runs", json={"name": "borealis", "tenant_id": "tenant-b"}).json()

    record = client.get(f"/runs/{first['run_id']}").json()
    assert record["tenant_id"] == "tenant-a"

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            },
            "steps": 1,
            "learning_rate": 0.001,
        },
    )

    assert response.status_code == 400
