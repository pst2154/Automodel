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

# Tinker-like AutoModel Prototype

This prototype has two layers. The original in-process `ServiceClient` mirrors
the smallest useful part of Tinker's public training loop while staying inside
NeMo AutoModel:

1. Create a `ServiceClient`.
2. Create a LoRA `TrainingClient`.
3. Call `forward_backward(data, "cross_entropy")`.
4. Call `optim_step(AdamParams(...))`.
5. Save adapter state and sample from the in-memory model.

The newer mixed-LoRA HTTP service keeps multiple LoRA adapters resident over
one base model, batches different adapters together, supports RL-style losses,
and exposes durable run/job metadata. Use the HTTP service for the
Tinker-like/mLoRA experiments; the original `ServiceClient` is now mostly a
small compatibility smoke-test path.

## What Works

- One resident base model per `ServiceClient` worker key in the legacy path.
- Multiple resident LoRA adapters over one shared base model in the mixed
  service path.
- Separate adapter IDs, adapter weights, optimizer state, and checkpoint output.
- Tinker-style synchronous futures via `.result()`.
- Cross-entropy SFT data using `Datum`, `ModelInput`, `target_tokens`, and `weights`.
- Mixed-service RL losses: `importance_sampling`, `ppo`, `cispo`, and `dro`.
- Mixed-adapter backends: `loop`, `grouped`, `triton`, and `grouped_triton`.
- SQLite-backed run/job/idempotency metadata plus interrupted-job continuation.
- Optional local worker-process supervision via `--worker-processes`.
- Durable worker placement metadata on each run when worker processes are
  configured.
- Worker management IPC via `POST /workers/{worker_id}/ping`, proving the HTTP
  service can route commands into a specific supervised process.
- Tiny GPU smoke tests in the `nvcr.io/nvidia/nemo-automodel:26.04` container.

## Current Limits

- The legacy `ServiceClient` path still serializes adapter execution with a
  lock, swaps adapter weights through one patched model, and only implements
  `cross_entropy`.
- The mixed service has batched adapter execution, RL losses, and durable
  metadata, but GPU work is still owned by a single in-process
  `MixedLoraServiceClient`.
- `--worker-processes` supervises local worker processes and exposes health
  endpoints. Runs receive stable worker placement metadata and management
  commands route into those processes, but GPU request execution still needs to
  move into the worker RPC protocol.
- Live Tinker parity is opt-in because it requires Tinker credentials and can
  consume hosted training quota.
- `--force-hf` is useful for arbitrary Hugging Face smoke-test models; AutoModel-native
  loading should be used for supported production targets.

## Files

- `nemo_automodel/services/tinker_api/client.py` contains the shared-base worker,
  training client, sampler, and adapter swapping logic.
- `nemo_automodel/services/tinker_api/types.py` contains the Tinker-like request
  and response dataclasses.
- `examples/tinker_api/prototype_sft.py` trains one LoRA adapter.
- `examples/tinker_api/multi_lora_prototype.py` trains two LoRA adapters over one
  shared base model.
- `tests/unit_tests/services/test_tinker_api.py` covers batching and worker reuse.

## H200 Scratch Setup

On `4u8g-gen-0277`, use `/home/scratch.asteiner` for persistent files because
the network home directory is small.

```bash
ssh 4u8g-gen-0277
mkdir -p /home/scratch.asteiner/{automodel,hf,checkpoints,data}
cd /home/scratch.asteiner/automodel

docker run --gpus all --rm -it --shm-size=64g \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -e HF_HOME=/home/scratch.asteiner/hf \
  nvcr.io/nvidia/nemo-automodel:26.04
```

Inside the container, run:

```bash
cd /home/scratch.asteiner/automodel
python examples/tinker_api/prototype_sft.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 5
```

For a tiny smoke test that avoids large checkpoint downloads, use:

```bash
python examples/tinker_api/prototype_sft.py \
  --base-model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 1 \
  --force-hf
```

To exercise two virtual LoRA adapters sharing one base model:

```bash
python examples/tinker_api/multi_lora_prototype.py \
  --base-model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --force-hf
```

Expected output includes:

```text
shared_worker=True
adapter_1=adapter-...
adapter_2=adapter-...
adapter_1_loss=...
adapter_2_loss=...
```

To train two Qwen LoRA adapters long enough to learn separate toy tasks:

```bash
python examples/tinker_api/train_two_lora_tasks.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 250 \
  --batch-size 4 \
  --lr 1e-3 \
  --rank 16 \
  --force-hf
```

On `4u8g-gen-0277`, this run produced `shared_worker=True`, drove both
adapter eval losses to approximately zero, and saved:

```text
/home/scratch.asteiner/checkpoints/qwen-two-lora-atlas
/home/scratch.asteiner/checkpoints/qwen-two-lora-borealis
```

## Experimental Mixed-Batch MultiLoRA

The serialized prototype above proves the Tinker-style control plane. The
experimental mixed-batch path in `mixed_client.py` is closer to mLoRA: multiple
adapters are resident in the model at the same time, and one concatenated batch
routes row ranges to different adapters during the same forward/backward pass.

```bash
python examples/tinker_api/mixed_lora_qwen.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 160 \
  --batch-size 2 \
  --lr 1e-3 \
  --rank 16
```

On `4u8g-gen-0277`, this produced:

```text
mixed_batch=True
atlas_loss before=6.3113 first_step=6.1240 last_step=0.0000 after=0.0000
borealis_loss before=5.8499 first_step=5.9166 last_step=0.0000 after=0.0000
/home/scratch.asteiner/checkpoints/mixed-qwen-atlas
/home/scratch.asteiner/checkpoints/mixed-qwen-borealis
```

This mixed path is intentionally single-node and pure PyTorch. It is useful for
validating the runtime shape, but production throughput would still need grouped
or fused LoRA kernels and distributed-aware adapter sharding.

## HTTP API Prototype

The experimental branch also includes a thin FastAPI wrapper around the
single-process mixed-LoRA worker. It is intentionally single-node, but it now
has the first service-shaped pieces: a worker queue that serializes GPU
operations, JSON-backed run metadata, checkpoint restore, and a compact HTTP
contract:

```text
GET  /health
POST /runs
GET  /runs
GET  /runs/{run_id}
POST /runs/{run_id}/forward_backward
POST /mixed_forward_backward
POST /runs/{run_id}/optim_step
POST /runs/{run_id}/save
POST /runs/{run_id}/sample
POST /train_steps
GET  /jobs
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
```

Run records include `status`, `sequence`, `optimizer_steps`,
`tenant_id`, `forward_backward_calls`, `last_loss`, `last_metrics`,
`last_checkpoint_path`, `last_error`, `restored_from`, `created_at`, and
`updated_at`. State-changing endpoints return both the compact run record and
the operation output, so clients do not need to make a second call after every
training step. Metadata is persisted to SQLite by default at
`$SCRATCH/tinker_api/metadata.sqlite3`. After a server restart, previously
known runs are listed as `detached` by default; if they have a checkpoint path,
`--restore-runs-on-startup` rehydrates them as resident adapters under the same
run ids. For debugging, the server can still use the old JSON files with
`--metadata-backend json`.

Mutating endpoints that can safely be retried accept an optional
`idempotency_key`: `POST /runs`, `POST /train_steps`,
`POST /runs/{run_id}/optim_step`, and `POST /runs/{run_id}/save`. Reusing the
same key with the same payload returns the original response; reusing it with a
different payload fails with `409`.

For a shared scratch server, start with a bearer token and a resident-adapter
cap:

```bash
TINKER_API_KEY=dev-secret \
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --mixed-lora-backend grouped \
  --max-resident-adapters 8 \
  --max-runs-per-tenant 2 \
  --tenant-rate-limit-per-minute 120 \
  --restore-runs-on-startup \
  --resume-interrupted-jobs-on-startup
```

Clients then pass `--api-key dev-secret`. Runs and `POST /train_steps` can also
carry a `tenant_id`; one training job may not mix runs from different tenants.
The server enforces both the global resident-adapter cap and the optional
per-tenant resident-run and requests-per-minute caps. Requests without a
`tenant_id` share a `_default` tenant bucket.

If the service restarts while an async `POST /train_steps` job is queued or
running, `--resume-interrupted-jobs-on-startup` requeues persisted training jobs
from their last completed step after resident runs have been restored. This is
step-boundary continuation, not mid-forward/backward checkpointing.

Start the server:

```bash
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --rank 16 \
  --host 127.0.0.1 \
  --port 18080
```

In another shell, run a small client smoke:

```bash
python examples/tinker_api/api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --base-model Qwen/Qwen3-0.6B \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 20 \
  --batch-size 1 \
  --tenant-id smoke
```

For the stronger Qwen learn-and-sample check, run:

```bash
python examples/tinker_api/api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --base-model Qwen/Qwen3-0.6B \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 160 \
  --batch-size 2 \
  --lr 1e-3 \
  --max-new-tokens 12 \
  --verify-samples
```

That client creates two runs, sends different Atlas and Borealis training
examples through one `POST /mixed_forward_backward` call per step, steps each
adapter independently, samples both adapters, verifies the expected route
strings, and saves separate checkpoints.

To let the service own the whole training loop through one `POST /train_steps`
job, add `--server-train-steps`:

```bash
python examples/tinker_api/api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --base-model Qwen/Qwen3-0.6B \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 120 \
  --batch-size 2 \
  --lr 1e-3 \
  --max-new-tokens 12 \
  --server-train-steps \
  --tenant-id smoke \
  --verify-samples
```

`POST /train_steps` supports `run_async: true`; in that mode it returns a job
record immediately and clients can poll `GET /jobs/{job_id}`. Cancel requests
are best-effort in this prototype: queued jobs become `canceled`, and running
jobs switch to `canceling` and stop at the next step boundary.

To restore those saved adapters in a fresh server process:

```bash
python examples/tinker_api/api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --base-model Qwen/Qwen3-0.6B \
  --cache-dir /home/scratch.asteiner/hf \
  --steps 0 \
  --atlas-checkpoint /home/scratch.asteiner/checkpoints/api-smoke-atlas \
  --borealis-checkpoint /home/scratch.asteiner/checkpoints/api-smoke-borealis \
  --verify-samples
```

Restore validates the checkpoint before loading weights. The saved
`adapter_config.json` must match the running service's base model, LoRA rank,
alpha, and target modules, and the adapter tensor keys must match the resident
mixed-LoRA layout.

This API layer is not production hardened. It has only simple bearer-token auth,
no distributed worker management yet, and only lightweight SQLite metadata. Its
purpose is to freeze the basic Tinker-like HTTP contract around the mixed
training worker while keeping the implementation pure Python.

### Mixed-LoRA Backends

The prototype now has four backend modes for the adapter delta inside each
patched linear layer:

- `loop`: original PyTorch implementation, one adapter range at a time.
- `grouped`: vectorized PyTorch reference path that stacks selected adapter
  weights and computes all active row deltas in one batched operation.
- `triton`: AutoModel's existing PEFT `LoRATritonFunction`, called once per
  active adapter range.
- `grouped_triton`: experimental grouped mixed-adapter Triton path that takes
  one adapter id per active batch row and launches across the mixed batch.

Use the grouped reference path when you want to mimic the shape of an mLoRA
grouped kernel without compiling a new kernel:

```bash
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --rank 16 \
  --mixed-lora-backend grouped
```

Use the Triton bridge when you want the existing compiled LoRA kernels:

```bash
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --rank 16 \
  --mixed-lora-backend triton
```

The `triton` path covers the LoRA delta for one adapter range at a time and
includes backward. The `grouped` path is closer to the desired production
kernel contract, but it still relies on PyTorch batched operations and dynamic
weight stacking.

Use the grouped Triton path when you want to exercise the first real
mixed-adapter kernel surface:

```bash
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /home/scratch.asteiner \
  --cache-dir /home/scratch.asteiner/hf \
  --rank 16 \
  --mixed-lora-backend grouped_triton
```

`grouped_triton` currently uses Triton for the LoRA forward delta, input
gradient (`dX`), and adapter weight gradients (`dA`/`dB`). The adapter-gradient
kernels use segmented reductions by adapter id, with each program owning an
adapter/rank/tile output region to avoid cross-program atomics in this
prototype.

### RL Losses

`forward_backward`, `mixed_forward_backward`, and `train_steps` accept these
`loss_fn` values:

- `cross_entropy`: SFT-style token cross entropy.
- `importance_sampling`: policy-gradient style `-ratio * advantage`.
- `ppo`: Tinker-style clipped-ratio objective.
- `cispo`: clipped-ratio gradient coefficient on target logprobs.
- `dro`: Tinker-style quadratic penalty on policy divergence.

The RL modes require Tinker-style `loss_fn_inputs["logprobs"]` and
`loss_fn_inputs["advantages"]`; `weights` remains an optional token mask or
weight. `loss_fn_config` supports `clip_low_threshold` and
`clip_high_threshold` for `ppo`/`cispo`, and `beta` for `dro`. Losses are
summed over tokens to match Tinker diagnostics.

### Live Tinker Parity

The repository includes an opt-in live parity harness at
`tests/integration_tests/services/test_tinker_live_parity.py`. It is skipped by
default because it needs Tinker credentials and runs against hosted
infrastructure.

Create or refresh a golden response:

```bash
RUN_TINKER_LIVE_PARITY=1 \
TINKER_API_KEY=... \
TINKER_UPDATE_GOLDEN=1 \
python -m pytest tests/integration_tests/services/test_tinker_live_parity.py
```

Compare future responses to the saved golden:

```bash
RUN_TINKER_LIVE_PARITY=1 \
TINKER_API_KEY=... \
python -m pytest tests/integration_tests/services/test_tinker_live_parity.py
```

The production kernel work that remains is:

1. Benchmarks comparing `loop`, `grouped`, `triton`, and `grouped_triton` on
   realistic Qwen hidden sizes, rank 16/32, and mixed tenant batch shapes.
2. Tuning grouped Triton tile sizes and, if needed, replacing serial
   per-output-region reductions with atomic split reductions for large batches.
3. Optional fused optimizer updates for many small adapter tensors, once the
   training service has enough resident-adapter churn to make Python optimizer
   overhead visible.

No attention, RoPE, base-model GEMM, or normalization kernels need to change
for the single-node prototype. The kernel work is specifically in the LoRA
delta path and, later, the small-adapter optimizer path.

## Next Steps

1. Move GPU request execution into the assigned supervised worker process
   instead of the in-process mixed-LoRA worker.
2. Benchmark and tune `grouped_triton` against the existing LoRA backends.
3. Run the opt-in live Tinker parity test whenever credentials/quota are
   available and check in reviewed golden values.
