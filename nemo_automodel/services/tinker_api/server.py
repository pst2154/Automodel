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
import os
import pathlib
import queue
import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Optional

from nemo_automodel.services.tinker_api.mixed_client import (
    MixedLoraBackend,
    MixedLoraServiceClient,
    MixedLoraTrainingClient,
)
from nemo_automodel.services.tinker_api.types import AdamParams, Datum, LoraConfig, ModelInput, SamplingParams
from nemo_automodel.services.tinker_api.worker_manager import ProcessWorkerManager, WorkerProcessRecord
from nemo_automodel.shared.import_utils import safe_import_from

HAS_FASTAPI, FastAPI = safe_import_from("fastapi", "FastAPI")
_, HTTPException = safe_import_from("fastapi", "HTTPException")
_, JSONResponse = safe_import_from("fastapi.responses", "JSONResponse")
HAS_PYDANTIC, BaseModel = safe_import_from("pydantic", "BaseModel")
_, Field = safe_import_from("pydantic", "Field")

MetadataBackend = Literal["sqlite", "json"]

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
    tenant_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class CreateRunResponse(BaseModel):
    """Created run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    status: str
    sequence: int
    worker_id: Optional[str] = None


class ForwardBackwardRequest(BaseModel):
    """Forward/backward request for one run."""

    data: list[DatumRequest]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)


class MixedForwardBackwardRequest(BaseModel):
    """Forward/backward request containing batches for multiple runs."""

    batches: dict[str, list[DatumRequest]]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)


class OptimStepRequest(BaseModel):
    """Optimizer step request."""

    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    idempotency_key: Optional[str] = None


class SaveRequest(BaseModel):
    """Save request."""

    name: str
    idempotency_key: Optional[str] = None


class SaveAndDetachRequest(BaseModel):
    """Save-and-detach request."""

    name: str
    idempotency_key: Optional[str] = None


class DetachRunRequest(BaseModel):
    """Detach request."""

    idempotency_key: Optional[str] = None


class SampleRequest(BaseModel):
    """Sampling request."""

    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class TrainStepsRequest(BaseModel):
    """Server-owned mixed training loop request."""

    batches: dict[str, list[DatumRequest]]
    steps: int
    learning_rate: float
    batch_size: int = 1
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    save_names: dict[str, str] = Field(default_factory=dict)
    run_async: bool = False
    tenant_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class RunRecord(BaseModel):
    """In-memory run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    status: str = "created"
    sequence: int = 0
    optimizer_steps: int = 0
    forward_backward_calls: int = 0
    last_loss: Optional[float] = None
    last_metrics: dict[str, Any] = Field(default_factory=dict)
    last_checkpoint_path: Optional[str] = None
    last_error: Optional[str] = None
    restored_from: Optional[str] = None
    worker_id: Optional[str] = None
    created_at: str
    updated_at: str


class JobRecord(BaseModel):
    """Queued operation metadata."""

    job_id: str
    kind: str
    status: str
    tenant_id: Optional[str] = None
    sequence: int = 0
    run_ids: list[str] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class IdempotencyRecord(BaseModel):
    """Stored response for a retryable mutating request."""

    key: str
    operation: str
    fingerprint: str
    status: str
    response: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class JobSubmitResponse(BaseModel):
    """Response returned when work is submitted asynchronously."""

    job: JobRecord


class TrainStepsResponse(BaseModel):
    """Synchronous server-owned train loop response."""

    job: JobRecord
    runs: dict[str, RunRecord]
    outputs: dict[str, Any]


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


class DetachRunResponse(BaseModel):
    """Detach response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class SaveAndDetachResponse(BaseModel):
    """Save-and-detach response with run metadata."""

    run: RunRecord
    save_output: dict[str, Any]
    detach_output: dict[str, Any]


class SampleResponseModel(BaseModel):
    """Sample response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class WorkerCommandResponse(BaseModel):
    """Response from one supervised worker command."""

    worker: WorkerProcessRecord
    result: dict[str, Any]


class WorkerEchoRequest(BaseModel):
    """Payload for testing worker RPC serialization."""

    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerRunsResponse(BaseModel):
    """Runs currently attached to one supervised worker."""

    worker: WorkerProcessRecord
    runs: list[dict[str, Any]] = Field(default_factory=list)
    assigned_run_count: int = 0


class WorkerOperationsResponse(BaseModel):
    """Model-operation RPC envelopes recorded by one supervised worker."""

    worker: WorkerProcessRecord
    operations: list[dict[str, Any]] = Field(default_factory=list)
    operation_count: int = 0


class WorkerReconcileResponse(BaseModel):
    """Result from reconciling resident runs with supervised workers."""

    reassigned_run_ids: list[str] = Field(default_factory=list)
    reattached_run_ids: list[str] = Field(default_factory=list)
    workers: list[WorkerProcessRecord] = Field(default_factory=list)


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


def _fingerprint_request(operation: str, request: BaseModel) -> str:
    payload = {"operation": operation, "request": _model_to_dict(request)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


class JsonStore:
    """Small JSON-backed metadata store for the prototype service."""

    def __init__(self, path: str | pathlib.Path, key: str, record_type):
        self.path = pathlib.Path(path)
        self.key = key
        self.record_type = record_type
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load known records from disk."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {record_id: self.record_type(**record) for record_id, record in payload.get(self.key, {}).items()}

    def save(self, records: dict[str, Any]) -> None:
        """Persist records atomically."""
        payload = {self.key: {record_id: _model_to_dict(record) for record_id, record in records.items()}}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


class SQLiteStore:
    """SQLite-backed metadata store for service records."""

    def __init__(self, path: str | pathlib.Path, key: str, record_type):
        self.path = pathlib.Path(path)
        self.key = key
        self.record_type = record_type
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    namespace TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, record_id)
                )
                """
            )

    def load(self) -> dict[str, Any]:
        """Load known records from SQLite."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_id, payload FROM records WHERE namespace = ? ORDER BY record_id",
                (self.key,),
            ).fetchall()
        return {record_id: self.record_type(**json.loads(payload)) for record_id, payload in rows}

    def save(self, records: dict[str, Any]) -> None:
        """Persist records in one transaction."""
        rows = [
            (self.key, record_id, json.dumps(_model_to_dict(record), sort_keys=True), _utc_now())
            for record_id, record in records.items()
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM records WHERE namespace = ?", (self.key,))
            conn.executemany(
                "INSERT INTO records(namespace, record_id, payload, updated_at) VALUES (?, ?, ?, ?)",
                rows,
            )


def _build_store(
    *,
    scratch_dir: str,
    backend: MetadataBackend,
    key: str,
    record_type,
):
    if backend == "sqlite":
        return SQLiteStore(pathlib.Path(scratch_dir) / "tinker_api" / "metadata.sqlite3", key, record_type)
    if backend == "json":
        return JsonStore(pathlib.Path(scratch_dir) / "tinker_api" / f"{key}.json", key, record_type)
    raise ValueError(f"Unknown metadata backend: {backend}")


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
    attn_implementation: str = "sdpa",
    target_modules: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    max_resident_adapters: Optional[int] = None,
    max_runs_per_tenant: Optional[int] = None,
    tenant_rate_limit_per_minute: Optional[int] = None,
    mixed_lora_backend: MixedLoraBackend = "loop",
    use_triton_lora: bool = False,
    metadata_backend: MetadataBackend = "sqlite",
    restore_runs_on_startup: bool = False,
    resume_interrupted_jobs_on_startup: bool = False,
    worker_processes: int = 0,
) -> FastAPI:
    """Create a single-process mixed-LoRA FastAPI app."""
    if not HAS_FASTAPI or not HAS_PYDANTIC:
        raise ImportError("The Tinker API server requires `fastapi` and `pydantic`. Install the service extras first.")
    if metadata_backend not in {"sqlite", "json"}:
        raise ValueError(f"Unknown metadata backend: {metadata_backend}")
    if max_runs_per_tenant is not None and max_runs_per_tenant <= 0:
        raise ValueError("max_runs_per_tenant must be positive")
    if tenant_rate_limit_per_minute is not None and tenant_rate_limit_per_minute <= 0:
        raise ValueError("tenant_rate_limit_per_minute must be positive")
    if worker_processes < 0:
        raise ValueError("worker_processes must be non-negative")

    service = MixedLoraServiceClient(
        base_model=base_model,
        scratch_dir=scratch_dir,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        lora_config=LoraConfig(rank=rank, alpha=alpha, target_modules=target_modules or []),
        mixed_lora_backend=mixed_lora_backend,
        use_triton_lora=use_triton_lora,
    )
    active_mixed_lora_backend = "triton" if use_triton_lora else mixed_lora_backend
    run_store = _build_store(scratch_dir=scratch_dir, backend=metadata_backend, key="runs", record_type=RunRecord)
    job_store = _build_store(scratch_dir=scratch_dir, backend=metadata_backend, key="jobs", record_type=JobRecord)
    idempotency_store = _build_store(
        scratch_dir=scratch_dir,
        backend=metadata_backend,
        key="idempotency",
        record_type=IdempotencyRecord,
    )
    records: dict[str, RunRecord] = run_store.load()
    runs: dict[str, MixedLoraTrainingClient] = {}
    for run_id, record in records.items():
        if record.status in {"failed", "detached"}:
            continue
        checkpoint_path = record.last_checkpoint_path or record.restored_from
        if restore_runs_on_startup and checkpoint_path:
            try:
                runs[run_id] = service.create_lora_training_client(
                    adapter_id=record.adapter_id,
                    checkpoint_path=checkpoint_path,
                )
                record.status = "ready"
                record.optimizer_steps = _client_step(runs[run_id])
                record.restored_from = checkpoint_path
                record.last_error = None
                record.updated_at = _utc_now()
                continue
            except Exception as exc:
                record.last_error = f"Startup restore failed: {type(exc).__name__}: {exc}"
        record.status = "detached"
        record.updated_at = _utc_now()
    if records:
        run_store.save(records)
    jobs: dict[str, JobRecord] = job_store.load()
    for job in jobs.values():
        if job.status in {"queued", "running", "canceling"}:
            request_payload = job.progress.get("request") if isinstance(job.progress, dict) else None
            can_resume = (
                resume_interrupted_jobs_on_startup
                and restore_runs_on_startup
                and job.kind == "train_steps"
                and request_payload is not None
            )
            if can_resume:
                job.status = "queued"
                job.error = None
                job.updated_at = _utc_now()
                continue
            job.status = "failed"
            job.error = "Job was interrupted by service restart"
            job.updated_at = _utc_now()
    if jobs:
        job_store.save(jobs)
    idempotency_records: dict[str, IdempotencyRecord] = idempotency_store.load()
    for idem_record in idempotency_records.values():
        if idem_record.status == "running":
            idem_record.status = "failed"
            idem_record.error = "Request was interrupted by service restart"
            idem_record.updated_at = _utc_now()
    if idempotency_records:
        idempotency_store.save(idempotency_records)
    records_lock = threading.RLock()
    jobs_lock = threading.RLock()
    idempotency_lock = threading.RLock()
    rate_limit_lock = threading.RLock()
    tenant_request_times: dict[str, deque[float]] = {}
    executor = QueuedExecutor()
    worker_manager = ProcessWorkerManager(num_workers=worker_processes) if worker_processes else None
    if worker_manager is not None:
        worker_manager.start()

    @asynccontextmanager
    async def lifespan(app):
        yield
        if worker_manager is not None:
            worker_manager.stop()

    app = FastAPI(title="NeMo AutoModel Tinker API Prototype", version="0.1.0", lifespan=lifespan)
    expected_api_key = api_key or os.environ.get("TINKER_API_KEY")

    if expected_api_key:

        @app.middleware("http")
        async def require_bearer_token(request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            authorization = request.headers.get("authorization")
            expected_authorization = f"Bearer {expected_api_key}"
            if authorization != expected_authorization:
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid bearer token"})
            return await call_next(request)

    def get_run(run_id: str) -> MixedLoraTrainingClient:
        client = runs.get(run_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return client

    def get_worker_record(worker_id: str) -> WorkerProcessRecord:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        records_by_id = {record.worker_id: record for record in worker_manager.snapshot()}
        worker = records_by_id.get(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Unknown worker_id: {worker_id}")
        return worker

    def attach_run_to_worker(record: RunRecord, *, reason: Optional[str] = None) -> None:
        if worker_manager is None or record.worker_id is None:
            return
        worker_manager.attach_run(record.worker_id, _model_to_dict(record))
        if reason is not None:
            worker_manager.record_operation(
                record.worker_id,
                operation="reattach_run",
                run_ids=[record.run_id],
                payload={"reason": reason},
            )

    def detach_run_from_worker(record: RunRecord) -> None:
        if worker_manager is None or record.worker_id is None:
            return
        worker_manager.detach_run(record.worker_id, record.run_id)

    def reattach_runs_to_worker(worker_id: str, *, reason: str) -> list[str]:
        if worker_manager is None:
            return []
        with records_lock:
            assigned_records = [
                record for run_id, record in records.items() if run_id in runs and record.worker_id == worker_id
            ]
        for record in assigned_records:
            attach_run_to_worker(record, reason=reason)
        return [record.run_id for record in assigned_records]

    def reattach_runs_to_workers(worker_ids: list[str], *, reason: str) -> list[str]:
        reattached_run_ids = []
        for worker_id in worker_ids:
            reattached_run_ids.extend(reattach_runs_to_worker(worker_id, reason=reason))
        return reattached_run_ids

    def reconcile_worker_assignments(*, reason: str) -> WorkerReconcileResponse:
        if worker_manager is None:
            return WorkerReconcileResponse()
        worker_records = worker_manager.snapshot()
        running_worker_ids = {record.worker_id for record in worker_records if record.status == "running"}
        reassigned_run_ids = []
        dirty_records = False
        with records_lock:
            resident_records = [record for run_id, record in records.items() if run_id in runs]
            for record in resident_records:
                if record.worker_id in running_worker_ids:
                    continue
                assigned_worker = worker_manager.assign(record.run_id)
                if assigned_worker is None:
                    continue
                record.worker_id = assigned_worker.worker_id
                record.updated_at = _utc_now()
                reassigned_run_ids.append(record.run_id)
                dirty_records = True
            if dirty_records:
                run_store.save(records)
            worker_ids = sorted({record.worker_id for record in resident_records if record.worker_id is not None})
        reattached_run_ids = reattach_runs_to_workers(worker_ids, reason=reason)
        return WorkerReconcileResponse(
            reassigned_run_ids=reassigned_run_ids,
            reattached_run_ids=reattached_run_ids,
            workers=worker_manager.snapshot(),
        )

    def record_worker_operation(operation: str, run_ids: list[str], payload: Optional[dict[str, Any]] = None) -> None:
        if worker_manager is None:
            return
        worker_run_ids: dict[str, list[str]] = {}
        for run_id in run_ids:
            record = get_record(run_id)
            if record.worker_id is None:
                continue
            worker_run_ids.setdefault(record.worker_id, []).append(run_id)
        for worker_id, assigned_run_ids in worker_run_ids.items():
            worker_manager.record_operation(
                worker_id,
                operation=operation,
                run_ids=assigned_run_ids,
                payload=payload,
            )

    if worker_manager is not None:
        with records_lock:
            dirty_records = False
            for run_id, client in runs.items():
                record = records[run_id]
                if record.worker_id is None:
                    assigned_worker = worker_manager.assign(run_id)
                    if assigned_worker is not None:
                        record.worker_id = assigned_worker.worker_id
                        record.adapter_id = client.adapter_id
                        record.updated_at = _utc_now()
                        dirty_records = True
                attach_run_to_worker(record, reason="startup")
            if dirty_records:
                run_store.save(records)

    app.state.executor = executor
    app.state.worker_manager = worker_manager
    app.state.reattach_runs_to_workers = reattach_runs_to_workers
    app.state.reconcile_worker_assignments = reconcile_worker_assignments

    @app.get("/health")
    def health() -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        stale_worker_run_ids = []
        worker_records = worker_manager.snapshot() if worker_manager is not None else []
        running_worker_ids = {record.worker_id for record in worker_records if record.status == "running"}
        with records_lock:
            for run_id, record in records.items():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
                if run_id in runs and worker_manager is not None and record.worker_id not in running_worker_ids:
                    stale_worker_run_ids.append(run_id)
        return {
            "status": "ok",
            "base_model": base_model,
            "num_runs": len(runs),
            "num_records": len(records),
            "mode": "mixed_lora_single_process",
            "model_execution": "api_process",
            "run_status_counts": status_counts,
            "stale_worker_run_ids": stale_worker_run_ids,
            "worker_assignment_ready": worker_manager is None or not stale_worker_run_ids,
            "queue_depth": executor.queue_depth(),
            "worker_alive": executor.is_alive(),
            "run_store": str(run_store.path),
            "job_store": str(job_store.path),
            "idempotency_store": str(idempotency_store.path),
            "auth_enabled": expected_api_key is not None,
            "max_resident_adapters": max_resident_adapters,
            "max_runs_per_tenant": max_runs_per_tenant,
            "tenant_rate_limit_per_minute": tenant_rate_limit_per_minute,
            "mixed_lora_backend": active_mixed_lora_backend,
            "use_triton_lora": use_triton_lora,
            "metadata_backend": metadata_backend,
            "restore_runs_on_startup": restore_runs_on_startup,
            "resume_interrupted_jobs_on_startup": resume_interrupted_jobs_on_startup,
            "worker_processes": worker_processes,
            "workers": [
                _model_to_dict(record) if isinstance(record, BaseModel) else asdict(record)
                for record in worker_records
            ],
        }

    @app.get("/workers", response_model=list[WorkerProcessRecord])
    def list_workers() -> list[WorkerProcessRecord]:
        if worker_manager is None:
            return []
        return worker_manager.snapshot()

    @app.post("/workers/restart_dead", response_model=list[WorkerProcessRecord])
    def restart_dead_workers() -> list[WorkerProcessRecord]:
        if worker_manager is None:
            return []
        restarted = worker_manager.restart_dead()
        reattach_runs_to_workers([record.worker_id for record in restarted], reason="worker_restart")
        return [get_worker_record(record.worker_id) for record in restarted]

    @app.post("/workers/reconcile", response_model=WorkerReconcileResponse)
    def reconcile_workers() -> WorkerReconcileResponse:
        if worker_manager is None:
            return WorkerReconcileResponse()
        return reconcile_worker_assignments(reason="manual_reconcile")

    @app.post("/workers/{worker_id}/ping", response_model=WorkerCommandResponse)
    def ping_worker(worker_id: str) -> WorkerCommandResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        result = worker_manager.submit(worker_id, "ping", timeout_seconds=5.0)
        return WorkerCommandResponse(worker=get_worker_record(worker_id), result=result)

    @app.post("/workers/{worker_id}/echo", response_model=WorkerCommandResponse)
    def echo_worker(worker_id: str, request: WorkerEchoRequest) -> WorkerCommandResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        result = worker_manager.submit(worker_id, "echo", payload=request.payload, timeout_seconds=5.0)
        return WorkerCommandResponse(worker=get_worker_record(worker_id), result=result)

    @app.get("/workers/{worker_id}/runs", response_model=WorkerRunsResponse)
    def list_worker_runs(worker_id: str) -> WorkerRunsResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        worker = get_worker_record(worker_id)
        result = worker_manager.list_runs(worker_id)
        return WorkerRunsResponse(
            worker=worker,
            runs=result.get("runs", []),
            assigned_run_count=result.get("assigned_run_count", 0),
        )

    @app.get("/workers/{worker_id}/operations", response_model=WorkerOperationsResponse)
    def list_worker_operations(worker_id: str) -> WorkerOperationsResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        worker = get_worker_record(worker_id)
        result = worker_manager.list_operations(worker_id)
        return WorkerOperationsResponse(
            worker=worker,
            operations=result.get("operations", []),
            operation_count=result.get("operation_count", 0),
        )

    def tenant_key(tenant_id: Optional[str]) -> str:
        return tenant_id or "_default"

    def enforce_tenant_rate_limit(tenant_id: Optional[str], operation: str) -> None:
        if tenant_rate_limit_per_minute is None:
            return
        key = tenant_key(tenant_id)
        now = time.monotonic()
        window_start = now - 60.0
        with rate_limit_lock:
            request_times = tenant_request_times.setdefault(key, deque())
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            if len(request_times) >= tenant_rate_limit_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Tenant {key!r} exceeded {tenant_rate_limit_per_minute} "
                        f"requests/minute for GPU operation {operation!r}"
                    ),
                )
            request_times.append(now)

    def enforce_tenant_run_quota(tenant_id: Optional[str]) -> None:
        if max_runs_per_tenant is None:
            return
        key = tenant_key(tenant_id)
        with records_lock:
            resident_count = sum(
                1
                for run_id, record in records.items()
                if run_id in runs and tenant_key(record.tenant_id) == key and record.status != "detached"
            )
        if resident_count >= max_runs_per_tenant:
            raise HTTPException(
                status_code=429,
                detail=f"Tenant {key!r} resident adapter capacity reached: {resident_count}/{max_runs_per_tenant}",
            )

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

    def tenant_for_runs(run_ids: list[str], requested_tenant_id: Optional[str]) -> Optional[str]:
        run_tenants = {get_record(run_id).tenant_id for run_id in run_ids}
        if len(run_tenants) > 1:
            raise HTTPException(status_code=400, detail="All runs in one job must belong to the same tenant")
        run_tenant_id = next(iter(run_tenants), None)
        if requested_tenant_id is not None and run_tenant_id is not None and requested_tenant_id != run_tenant_id:
            raise HTTPException(status_code=403, detail="Request tenant_id does not match run tenant_id")
        return requested_tenant_id or run_tenant_id

    def create_job(kind: str, run_ids: list[str], tenant_id: Optional[str]) -> JobRecord:
        now = _utc_now()
        job = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            kind=kind,
            status="queued",
            tenant_id=tenant_id,
            run_ids=run_ids,
            created_at=now,
            updated_at=now,
        )
        with jobs_lock:
            jobs[job.job_id] = job
            job_store.save(jobs)
        return job

    def get_job_record(job_id: str) -> JobRecord:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
            return job

    def mark_job(
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> JobRecord:
        with jobs_lock:
            job = get_job_record(job_id)
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.sequence += 1
            job.updated_at = _utc_now()
            job_store.save(jobs)
            return job

    def fail_job(job_id: str, exc: Exception) -> None:
        mark_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def get_idempotent_response(operation: str, key: Optional[str], request: BaseModel) -> Optional[dict[str, Any]]:
        if key is None:
            return None
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            record = idempotency_records.get(key)
            if record is None:
                now = _utc_now()
                idempotency_records[key] = IdempotencyRecord(
                    key=key,
                    operation=operation,
                    fingerprint=fingerprint,
                    status="running",
                    created_at=now,
                    updated_at=now,
                )
                idempotency_store.save(idempotency_records)
                return None
            if record.operation != operation or record.fingerprint != fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail=f"Idempotency key {key!r} was already used for a different request",
                )
            if record.status == "succeeded" and record.response is not None:
                return record.response
            if record.status == "failed":
                raise HTTPException(
                    status_code=409,
                    detail=f"Previous request for idempotency key {key!r} failed: {record.error}",
                )
            raise HTTPException(status_code=409, detail=f"Request for idempotency key {key!r} is still running")

    def store_idempotent_response(operation: str, key: Optional[str], request: BaseModel, response: Any) -> None:
        if key is None:
            return
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            idempotency_records[key] = IdempotencyRecord(
                key=key,
                operation=operation,
                fingerprint=fingerprint,
                status="succeeded",
                response=_model_to_dict(response) if isinstance(response, BaseModel) else response,
                created_at=idempotency_records[key].created_at,
                updated_at=_utc_now(),
            )
            idempotency_store.save(idempotency_records)

    def store_idempotent_error(operation: str, key: Optional[str], request: BaseModel, exc: Exception) -> None:
        if key is None:
            return
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            created_at = idempotency_records[key].created_at
            idempotency_records[key] = IdempotencyRecord(
                key=key,
                operation=operation,
                fingerprint=fingerprint,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                created_at=created_at,
                updated_at=_utc_now(),
            )
            idempotency_store.save(idempotency_records)

    def run_train_steps(request: TrainStepsRequest, job: JobRecord) -> TrainStepsResponse:
        run_ids = list(request.batches)
        if request.steps < 0:
            raise ValueError("steps must be non-negative")
        if request.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if get_job_record(job.job_id).status == "canceled":
            return TrainStepsResponse(job=job, runs={}, outputs={})

        clients_by_run = {run_id: get_run(run_id) for run_id in run_ids}
        batches_by_run = {
            run_id: [_datum_from_request(datum) for datum in batch] * request.batch_size
            for run_id, batch in request.batches.items()
        }
        adam_params = AdamParams(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        )

        outputs: dict[str, Any] = {}
        existing_progress = dict(get_job_record(job.job_id).progress)
        start_step = int(existing_progress.get("step", 0))
        first_losses = existing_progress.get("first_losses")
        last_losses = existing_progress.get("last_losses")
        progress_request = existing_progress.get("request", _model_to_dict(request))
        mark_job(
            job.job_id,
            status="running",
            progress={
                "step": start_step,
                "total_steps": request.steps,
                "request": progress_request,
                "first_losses": first_losses,
                "last_losses": last_losses,
            },
        )
        try:
            for step in range(start_step, request.steps):
                if get_job_record(job.job_id).status == "canceling":
                    mark_job(job.job_id, status="canceled", progress={"step": step, "total_steps": request.steps})
                    raise RuntimeError("Job canceled")
                adapter_batches = {clients_by_run[run_id].adapter_id: batch for run_id, batch in batches_by_run.items()}
                for run_id in run_ids:
                    mark_run(run_id, status="running")
                mixed_outputs = service.forward_backward_mixed(
                    adapter_batches,
                    request.loss_fn,
                    request.loss_fn_config,
                ).result()
                record_worker_operation(
                    "train_steps.forward_backward",
                    run_ids,
                    {"job_id": job.job_id, "step": step + 1, "loss_fn": request.loss_fn},
                )
                losses = {}
                for run_id, client in clients_by_run.items():
                    output = mixed_outputs[client.adapter_id]
                    record = mark_run(run_id, status="ready")
                    record.forward_backward_calls += 1
                    record.last_loss = output.loss
                    record.last_metrics = dict(output.metrics)
                    losses[run_id] = output.loss
                    outputs[run_id] = asdict(output)
                first_losses = first_losses or losses
                last_losses = losses
                for run_id, client in clients_by_run.items():
                    mark_run(run_id, status="optimizing")
                    step_output = client.optim_step(adam_params).result()
                    record_worker_operation(
                        "train_steps.optim_step",
                        [run_id],
                        {"job_id": job.job_id, "step": step + 1, "learning_rate": request.learning_rate},
                    )
                    record = mark_run(run_id, status="ready")
                    record.optimizer_steps = step_output.step
                    outputs[f"{run_id}:optim_step"] = asdict(step_output)
                mark_job(
                    job.job_id,
                    progress={
                        "step": step + 1,
                        "total_steps": request.steps,
                        "request": progress_request,
                        "first_losses": first_losses,
                        "last_losses": losses,
                    },
                )

            saved_paths = {}
            for run_id, save_name in request.save_names.items():
                client = clients_by_run[run_id]
                mark_run(run_id, status="saving")
                save_output = client.save_state(save_name).result()
                record_worker_operation("train_steps.save", [run_id], {"job_id": job.job_id, "name": save_name})
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = save_output.path
                saved_paths[run_id] = save_output.path
                outputs[f"{run_id}:save"] = asdict(save_output)

            result = {
                "first_losses": first_losses,
                "last_losses": last_losses,
                "saved_paths": saved_paths,
            }
            job = mark_job(job.job_id, status="succeeded", result=result)
            with records_lock:
                run_store.save(records)
                run_records = {run_id: records[run_id] for run_id in run_ids}
            return TrainStepsResponse(job=job, runs=run_records, outputs=outputs)
        except Exception as exc:
            if get_job_record(job.job_id).status != "canceled":
                fail_job(job.job_id, exc)
            for run_id in run_ids:
                if run_id in records:
                    mark_run_failed(run_id, exc)
            raise

    def resume_interrupted_train_jobs() -> None:
        if not resume_interrupted_jobs_on_startup:
            return
        for job in list(jobs.values()):
            if job.kind != "train_steps" or job.status != "queued":
                continue
            request_payload = job.progress.get("request") if isinstance(job.progress, dict) else None
            if request_payload is None:
                continue
            missing_runs = [run_id for run_id in job.run_ids if run_id not in runs]
            if missing_runs:
                mark_job(
                    job.job_id,
                    status="failed",
                    error=f"Cannot resume job; runs are not resident after restart: {missing_runs}",
                )
                continue
            request = TrainStepsRequest(**request_payload)
            executor.submit(lambda request=request, job=job: run_train_steps(request, job))

    resume_interrupted_train_jobs()

    @app.post("/runs", response_model=CreateRunResponse)
    def create_run(request: CreateRunRequest) -> CreateRunResponse:
        existing = get_idempotent_response("create_run", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(request.tenant_id, "create_run")

        def op() -> CreateRunResponse:
            try:
                if max_resident_adapters is not None and len(runs) >= max_resident_adapters:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Resident adapter capacity reached: {len(runs)}/{max_resident_adapters}",
                    )
                enforce_tenant_run_quota(request.tenant_id)
                run_id = f"run_{uuid.uuid4().hex[:12]}"
                assigned_worker = worker_manager.assign(run_id) if worker_manager is not None else None
                client = service.create_lora_training_client(
                    adapter_id=request.adapter_id,
                    checkpoint_path=request.checkpoint_path,
                )
                runs[run_id] = client
                now = _utc_now()
                record = RunRecord(
                    run_id=run_id,
                    adapter_id=client.adapter_id,
                    name=request.name,
                    tenant_id=request.tenant_id,
                    status="ready" if request.checkpoint_path else "created",
                    optimizer_steps=_client_step(client),
                    last_checkpoint_path=request.checkpoint_path,
                    restored_from=request.checkpoint_path,
                    worker_id=assigned_worker.worker_id if assigned_worker is not None else None,
                    created_at=now,
                    updated_at=now,
                )
                with records_lock:
                    records[run_id] = record
                    run_store.save(records)
                attach_run_to_worker(record)
                record_worker_operation(
                    "create_run",
                    [run_id],
                    {
                        "adapter_id": record.adapter_id,
                        "checkpoint_path": request.checkpoint_path,
                    },
                )
                response = CreateRunResponse(
                    run_id=run_id,
                    adapter_id=client.adapter_id,
                    name=request.name,
                    status=record.status,
                    sequence=record.sequence,
                    worker_id=record.worker_id,
                )
                store_idempotent_response("create_run", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error("create_run", request.idempotency_key, request, exc)
                raise

        return executor.submit(op).result()

    @app.get("/runs", response_model=list[RunRecord])
    def list_runs() -> list[RunRecord]:
        with records_lock:
            return list(records.values())

    @app.get("/runs/{run_id}", response_model=RunRecord)
    def get_run_record(run_id: str) -> RunRecord:
        return get_record(run_id)

    @app.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        with jobs_lock:
            return list(jobs.values())

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return get_job_record(job_id)

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        job = get_job_record(job_id)
        if job.status == "queued":
            return mark_job(job_id, status="canceled")
        if job.status == "running":
            return mark_job(job_id, status="canceling")
        return job

    @app.post("/train_steps")
    def train_steps(request: TrainStepsRequest) -> TrainStepsResponse | JobSubmitResponse:
        existing = get_idempotent_response("train_steps", request.idempotency_key, request)
        if existing is not None:
            return existing
        tenant_id = tenant_for_runs(list(request.batches), request.tenant_id)
        enforce_tenant_rate_limit(tenant_id, "train_steps")
        job = create_job("train_steps", list(request.batches), tenant_id)
        mark_job(
            job.job_id,
            progress={
                "step": 0,
                "total_steps": request.steps,
                "request": _model_to_dict(request),
            },
        )
        if request.run_async:
            future = executor.submit(lambda: run_train_steps(request, job))

            def complete_job(done_future: Future) -> None:
                try:
                    done_future.result()
                except Exception:
                    pass

            future.add_done_callback(complete_job)
            response = JobSubmitResponse(job=job)
            store_idempotent_response("train_steps", request.idempotency_key, request, response)
            return response
        try:
            response = executor.submit(lambda: run_train_steps(request, job)).result()
            store_idempotent_response("train_steps", request.idempotency_key, request, response)
            return response
        except Exception as exc:
            store_idempotent_error("train_steps", request.idempotency_key, request, exc)
            raise

    @app.post("/runs/{run_id}/forward_backward")
    def forward_backward(run_id: str, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "forward_backward")

        def op() -> ForwardBackwardResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="running")
                data = [_datum_from_request(datum) for datum in request.data]
                output = service.forward_backward_mixed(
                    {client.adapter_id: data},
                    request.loss_fn,
                    request.loss_fn_config,
                ).result()[client.adapter_id]
                record_worker_operation(
                    "forward_backward",
                    [run_id],
                    {"loss_fn": request.loss_fn, "batch_size": len(request.data)},
                )
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
        tenant_id = tenant_for_runs(list(request.batches), None)
        enforce_tenant_rate_limit(tenant_id, "mixed_forward_backward")

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
                outputs = service.forward_backward_mixed(
                    batches_by_adapter,
                    request.loss_fn,
                    request.loss_fn_config,
                ).result()
                record_worker_operation(
                    "mixed_forward_backward",
                    active_run_ids,
                    {"loss_fn": request.loss_fn, "num_runs": len(active_run_ids)},
                )
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
        existing = get_idempotent_response(f"optim_step:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "optim_step")

        def op() -> OptimStepResponseModel:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="optimizing")
                output = client.optim_step(_adam_from_request(request)).result()
                record_worker_operation("optim_step", [run_id], {"learning_rate": request.learning_rate})
                record = mark_run(run_id, status="ready")
                record.optimizer_steps = output.step
                run_store.save(records)
                response = OptimStepResponseModel(run=record, output=asdict(output))
                store_idempotent_response(f"optim_step:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"optim_step:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/save")
    def save(run_id: str, request: SaveRequest) -> SaveResponse:
        existing = get_idempotent_response(f"save:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "save")

        def op() -> SaveResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="saving")
                output = client.save_state(request.name).result()
                record_worker_operation("save", [run_id], {"name": request.name})
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = output.path
                run_store.save(records)
                response = SaveResponse(run=record, output=asdict(output))
                store_idempotent_response(f"save:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"save:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/detach")
    def detach_run(run_id: str, request: DetachRunRequest) -> DetachRunResponse:
        existing = get_idempotent_response(f"detach:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "detach")

        def op() -> DetachRunResponse:
            client = get_run(run_id)
            try:
                record = mark_run(run_id, status="detaching")
                output = client.detach().result()
                record_worker_operation("detach_run", [run_id], {"adapter_id": client.adapter_id})
                detach_run_from_worker(record)
                runs.pop(run_id, None)
                record = mark_run(run_id, status="detached")
                run_store.save(records)
                response = DetachRunResponse(run=record, output=asdict(output))
                store_idempotent_response(f"detach:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"detach:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/save_and_detach")
    def save_and_detach(run_id: str, request: SaveAndDetachRequest) -> SaveAndDetachResponse:
        existing = get_idempotent_response(f"save_and_detach:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "save_and_detach")

        def op() -> SaveAndDetachResponse:
            client = get_run(run_id)
            try:
                record = mark_run(run_id, status="saving")
                save_output = client.save_state(request.name).result()
                record.last_checkpoint_path = save_output.path
                record_worker_operation("save", [run_id], {"name": request.name})
                record = mark_run(run_id, status="detaching")
                detach_output = client.detach().result()
                record_worker_operation("detach_run", [run_id], {"adapter_id": client.adapter_id})
                detach_run_from_worker(record)
                runs.pop(run_id, None)
                record = mark_run(run_id, status="detached")
                run_store.save(records)
                response = SaveAndDetachResponse(
                    run=record,
                    save_output=asdict(save_output),
                    detach_output=asdict(detach_output),
                )
                store_idempotent_response(f"save_and_detach:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"save_and_detach:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    @app.post("/runs/{run_id}/sample")
    def sample(run_id: str, request: SampleRequest) -> SampleResponseModel:
        enforce_tenant_rate_limit(get_record(run_id).tenant_id, "sample")

        def op() -> SampleResponseModel:
            client = get_run(run_id)
            try:
                output = service.sample(client.adapter_id, request.prompt, _sampling_from_request(request)).result()
                record_worker_operation("sample", [run_id], {"max_new_tokens": request.max_new_tokens})
                record = mark_run(run_id, status="ready")
                return SampleResponseModel(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(op).result()

    return app
