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

import uuid
from dataclasses import asdict
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


class CreateRunResponse(BaseModel):
    """Created run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None


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
    runs: dict[str, MixedLoraTrainingClient] = {}
    records: dict[str, RunRecord] = {}

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
            "mode": "mixed_lora_single_process",
        }

    @app.post("/runs", response_model=CreateRunResponse)
    def create_run(request: CreateRunRequest) -> CreateRunResponse:
        client = service.create_lora_training_client()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        runs[run_id] = client
        records[run_id] = RunRecord(run_id=run_id, adapter_id=client.adapter_id, name=request.name)
        return CreateRunResponse(run_id=run_id, adapter_id=client.adapter_id, name=request.name)

    @app.get("/runs", response_model=list[RunRecord])
    def list_runs() -> list[RunRecord]:
        return list(records.values())

    @app.post("/runs/{run_id}/forward_backward")
    def forward_backward(run_id: str, request: ForwardBackwardRequest) -> dict[str, Any]:
        client = get_run(run_id)
        data = [_datum_from_request(datum) for datum in request.data]
        output = service.forward_backward_mixed({client.adapter_id: data}, request.loss_fn).result()[client.adapter_id]
        return asdict(output)

    @app.post("/mixed_forward_backward")
    def mixed_forward_backward(request: MixedForwardBackwardRequest) -> dict[str, Any]:
        batches_by_adapter = {}
        run_to_adapter = {}
        for run_id, batch in request.batches.items():
            client = get_run(run_id)
            batches_by_adapter[client.adapter_id] = [_datum_from_request(datum) for datum in batch]
            run_to_adapter[run_id] = client.adapter_id
        outputs = service.forward_backward_mixed(batches_by_adapter, request.loss_fn).result()
        return {run_id: asdict(outputs[adapter_id]) for run_id, adapter_id in run_to_adapter.items()}

    @app.post("/runs/{run_id}/optim_step")
    def optim_step(run_id: str, request: OptimStepRequest) -> dict[str, Any]:
        client = get_run(run_id)
        output = client.optim_step(_adam_from_request(request)).result()
        return asdict(output)

    @app.post("/runs/{run_id}/save")
    def save(run_id: str, request: SaveRequest) -> dict[str, Any]:
        client = get_run(run_id)
        output = client.save_state(request.name).result()
        return asdict(output)

    @app.post("/runs/{run_id}/sample")
    def sample(run_id: str, request: SampleRequest) -> dict[str, Any]:
        client = get_run(run_id)
        output = service.sample(client.adapter_id, request.prompt, _sampling_from_request(request)).result()
        return asdict(output)

    return app
