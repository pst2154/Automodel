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
import queue
import threading
import uuid
from concurrent.futures import Future
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from nemo_automodel.services.tinker_api.mixed_client import MixedLoraServiceClient, MixedLoraTrainingClient
from nemo_automodel.services.tinker_api.types import AdamParams, Datum, LoraConfig, ModelInput, SamplingParams
from nemo_automodel.shared.import_utils import safe_import_from

HAS_FASTAPI, FastAPI = safe_import_from("fastapi", "FastAPI")
_, HTTPException = safe_import_from("fastapi", "HTTPException")
HAS_PYDANTIC, BaseModel = safe_import_from("pydantic", "BaseModel")
_, Field = safe_import_from("pydantic", "Field")

if not HAS_PYDANTIC:  # pragma: no cover

    class BaseModel:
        """Placeholder used only to keep module import safe without pydantic."""

    def Field(*args, **kwargs):  # noqa: N802
        """Placeholder used only to keep module import safe without pydantic."""
        return None


class ModelInputRequest(BaseModel):
    """Token IDs for one model input sequence."""

    tokens: list[int]


class DatumRequest(BaseModel):
    """One tokenized training datum."""

    model_input: ModelInputRequest
    loss_fn_inputs: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    """Create one resident LoRA adapter run."""

    name: Optional[str] = None
    adapter_id: Optional[str] = None
    checkpoint_path: Optional[str] = None


class CreateRunResponse(BaseModel):
    """Created run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    status: str
    sequence: int


class ForwardBackwardRequest(BaseModel):
    """Forward/backward request for one run."""

    data: list[DatumRequest]
    loss_fn: str = "cross_entropy"


class MixedForwardBackwardRequest(BaseModel):
    """Forward/backward request containing batches for multiple runs."""

    batches: dict[str, list[DatumRequest]]
    loss_fn: str = "cross_entropy"


class OptimStepRequest(BaseModel):
    """Optimizer step request."""

    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class SaveRequest(BaseModel):
    """Save request."""

    name: str


class SampleRequest(BaseModel):
    """Sampling request."""

    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class RunRecord(BaseModel):
    """In-memory run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    status: str = "created"
    sequence: int = 0
    optimizer_steps: int = 0
    forward_backward_calls: int = 0
    last_loss: Optional[float] = None
    last_metrics: dict[str, float] = Field(default_factory=dict)
    last_checkpoint_path: Optional[str] = None
    last_error: Optional[str] = None
    restored_from: Optional[str] = None
    created_at: str
    updated_at: str


class ForwardBackwardResponse(BaseModel):
    """Forward/backward response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class OptimStepResponseModel(BaseModel):
    """Optimizer step response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class SaveResponse(BaseModel):
    """Save response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class SampleResponseModel(BaseModel):
    """Sample response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


def _datum_from_request(request: DatumRequest) -> Datum:
    loss_fn_inputs = dict(request.loss_fn_inputs)
    target_tokens = loss_fn_inputs.get("target_tokens")
    if isinstance(target_tokens, dict) and "tokens" in target_tokens:
        loss_fn_inputs["target_tokens"] = ModelInput.from_ints(target_tokens["tokens"])
    return Datum(
        model_input=ModelInput.from_ints(request.model_input.tokens),
        loss_fn_inputs=loss_fn_inputs,
    )


def _adam_from_request(request: OptimStepRequest) -> AdamParams:
    return AdamParams(
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        betas=request.betas,
        eps=request.eps,
    )


def _sampling_from_request(request: SampleRequest) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _client_step(client: MixedLoraTrainingClient) -> int:
    handle = getattr(client, "handle", None)
    return int(getattr(handle, "step", 0))


class RunStore:
    """Small JSON-backed run metadata store for the prototype service."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, RunRecord]:
        """Load known run records from disk."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {run_id: RunRecord(**record) for run_id, record in payload.get("runs", {}).items()}

    def save(self, records: dict[str, RunRecord]) -> None:
        """Persist run records atomically."""
        payload = {"runs": {run_id: _model_to_dict(record) for run_id, record in records.items()}}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


class QueuedExecutor:
    """One-thread executor that serializes all GPU-owned service operations."""

    def __init__(self):
        self._queue: queue.Queue[tuple[Future, Any]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="tinker-api-worker", daemon=True)
        self._thread.start()

    def submit(self, fn):
        """Run `fn` on the worker thread and return a Future."""
        future = Future()
        self._queue.put((future, fn))
        return future

    def queue_depth(self) -> int:
        """Return approximate queued operation count."""
        return self._queue.qsize()

    def is_alive(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._thread.is_alive()

    def _worker(self) -> None:
        while True:
            future, fn = self._queue.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except Exception as exc:
                    future.set_exception(exc)
            self._queue.task_done()


def create_app(
    *,
    base_model: str,
    scratch_dir: str = "/home/scratch.asteiner",
    cache_dir: Optional[str] = None,
    rank: int = 16,
    alpha: Optional[int] = None,
    device: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> FastAPI:
    """Create a single-process mixed-LoRA FastAPI app."""
    if not HAS_FASTAPI or not HAS_PYDANTIC:
        raise ImportError("The Tinker API server requires `fastapi` and `pydantic`. Install the service extras first.")

    service = MixedLoraServiceClient(
        base_model=base_model,
        scratch_dir=scratch_dir,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        lora_config=LoraConfig(rank=rank, alpha=alpha),
    )
    run_store = RunStore(pathlib.Path(scratch_dir) / "tinker_api" / "runs.json")
    records: dict[str, RunRecord] = run_store.load()
    for record in records.values():
        if record.status not in {"failed", "detached"}:
            record.status = "detached"
            record.updated_at = _utc_now()
    if records:
        run_store.save(records)
    runs: dict[str, MixedLoraTrainingClient] = {}
    records_lock = threading.RLock()
    executor = QueuedExecutor()

    app = FastAPI(title="NeMo AutoModel Tinker API Prototype", version="0.1.0")

    def get_run(run_id: str) -> MixedLoraTrainingClient:
        client = runs.get(run_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return client

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "base_model": base_model,
            "num_runs": len(runs),
            "num_records": len(records),
            "mode": "mixed_lora_single_process",
            "queue_depth": executor.queue_depth(),
            "worker_alive": executor.is_alive(),
            "run_store": str(run_store.path),
        }

    def get_record(run_id: str) -> RunRecord:
        with records_lock:
            record = records.get(run_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
            return record

    def mark_run(run_id: str, *, status: str, error: Optional[str] = None) -> RunRecord:
        with records_lock:
            record = get_record(run_id)
            record.status = status
            record.last_error = error
            record.sequence += 1
            record.updated_at = _utc_now()
            run_store.save(records)
            return record

    def mark_run_failed(run_id: str, exc: Exception) -> None:
        mark_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    @app.post("/runs", response_model=CreateRunResponse)
    def create_run(request: CreateRunRequest) -> CreateRunResponse:
        def op() -> CreateRunResponse:
            client = service.create_lora_training_client(
                adapter_id=request.adapter_id,
                checkpoint_path=request.checkpoint_path,
            )
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            runs[run_id] = client
            now = _utc_now()
            record = RunRecord(
                run_id=run_id,
                adapter_id=client.adapter_id,
                name=request.name,
                status="ready" if request.checkpoint_path else "created",
                optimizer_steps=_client_step(client),
                last_checkpoint_path=request.checkpoint_path,
                restored_from=request.checkpoint_path,
                created_at=now,
                updated_at=now,
            )
            with records_lock:
                records[run_id] = record
                run_store.save(records)
            return CreateRunResponse(
                run_id=run_id,
                adapter_id=client.adapter_id,
                name=request.name,
                status=record.status,
                sequence=record.sequence,
            )

        return executor.submit(op).result()

    @app.get("/runs", response_model=list[RunRecord])
    def list_runs() -> list[RunRecord]:
        with records_lock:
            return list(records.values())

    @app.get("/runs/{run_id}", response_model=RunRecord)
    def get_run_record(run_id: str) -> RunRecord:
        return get_record(run_id)

    @app.post("/runs/{run_id}/forward_backward")
    def forward_backward(run_id: str, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        def op() -> ForwardBackwardResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="running")
                data = [_datum_from_request(datum) for datum in request.data]
                output = service.forward_backward_mixed({client.adapter_id: data}, request.loss_fn).result()[
                    client.adapter_id
                ]
                record = mark_run(run_id, status="ready")
                record.forward_backward_calls += 1
                record.last_loss = output.loss
                record.last_metrics = dict(output.metrics)
                run_store.save(records)
                return ForwardBackwardResponse(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/mixed_forward_backward")
    def mixed_forward_backward(request: MixedForwardBackwardRequest) -> dict[str, ForwardBackwardResponse]:
        def op() -> dict[str, ForwardBackwardResponse]:
            batches_by_adapter = {}
            run_to_adapter = {}
            active_run_ids = list(request.batches)
            try:
                for run_id, batch in request.batches.items():
                    mark_run(run_id, status="running")
                    client = get_run(run_id)
                    batches_by_adapter[client.adapter_id] = [_datum_from_request(datum) for datum in batch]
                    run_to_adapter[run_id] = client.adapter_id
                outputs = service.forward_backward_mixed(batches_by_adapter, request.loss_fn).result()
                responses = {}
                for run_id, adapter_id in run_to_adapter.items():
                    output = outputs[adapter_id]
                    record = mark_run(run_id, status="ready")
                    record.forward_backward_calls += 1
                    record.last_loss = output.loss
                    record.last_metrics = dict(output.metrics)
                    run_store.save(records)
                    responses[run_id] = ForwardBackwardResponse(run=record, output=asdict(output))
                return responses
            except Exception as exc:
                for run_id in active_run_ids:
                    if run_id in records:
                        mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/optim_step")
    def optim_step(run_id: str, request: OptimStepRequest) -> OptimStepResponseModel:
        def op() -> OptimStepResponseModel:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="optimizing")
                output = client.optim_step(_adam_from_request(request)).result()
                record = mark_run(run_id, status="ready")
                record.optimizer_steps = output.step
                run_store.save(records)
                return OptimStepResponseModel(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/save")
    def save(run_id: str, request: SaveRequest) -> SaveResponse:
        def op() -> SaveResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="saving")
                output = client.save_state(request.name).result()
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = output.path
                run_store.save(records)
                return SaveResponse(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/sample")
    def sample(run_id: str, request: SampleRequest) -> SampleResponseModel:
        def op() -> SampleResponseModel:
            client = get_run(run_id)
            try:
                output = service.sample(client.adapter_id, request.prompt, _sampling_from_request(request)).result()
                record = mark_run(run_id, status="ready")
                return SampleResponseModel(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    return app
