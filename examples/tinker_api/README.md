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

This prototype is an in-process Python API that mirrors the smallest useful
part of Tinker's public training loop while staying inside NeMo AutoModel:

1. Create a `ServiceClient`.
2. Create a LoRA `TrainingClient`.
3. Call `forward_backward(data, "cross_entropy")`.
4. Call `optim_step(AdamParams(...))`.
5. Save adapter state and sample from the in-memory model.

The current version also supports multiple virtual LoRA adapters over one
resident base model when clients are created from the same `ServiceClient` with
the same base-model and LoRA configuration. Adapter calls are serialized and
swap adapter weights through the shared model; this proves the Tinker-like
control-plane semantics before adding mLoRA-style multi-adapter batching.

## What Works

- One resident base model per `ServiceClient` worker key.
- Multiple virtual LoRA adapters over that shared base model.
- Separate adapter IDs, adapter weights, optimizer state, and checkpoint output.
- Tinker-style synchronous futures via `.result()`.
- Cross-entropy SFT data using `Datum`, `ModelInput`, `target_tokens`, and `weights`.
- Tiny GPU smoke tests in the `nvcr.io/nvidia/nemo-automodel:26.04` container.

## Current Limits

- Adapter execution is serialized with a lock.
- Adapter weights are swapped through one patched model rather than batched together.
- Only `cross_entropy` is implemented.
- The core training worker is still in-process; the HTTP layer is a thin
  FastAPI wrapper without durable orchestration.
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
training step. Metadata is persisted to
`$SCRATCH/tinker_api/runs.json` and job metadata is persisted to
`$SCRATCH/tinker_api/jobs.json`; after a server restart, previously known runs
are listed as `detached` until a new resident adapter is created from a
checkpoint.

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
  --max-resident-adapters 8
```

Clients then pass `--api-key dev-secret`. Runs and `POST /train_steps` can also
carry a `tenant_id`; one training job may not mix runs from different tenants.

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
no SQL database, and no multi-process worker management yet. Its purpose is to
freeze the basic Tinker-like HTTP contract around the mixed training worker
while keeping the implementation pure Python.

### Mixed-LoRA Backends

The prototype now has three backend modes for the adapter delta inside each
patched linear layer:

- `loop`: original PyTorch implementation, one adapter range at a time.
- `grouped`: vectorized PyTorch reference path that stacks selected adapter
  weights and computes all active row deltas in one batched operation.
- `triton`: AutoModel's existing PEFT `LoRATritonFunction`, called once per
  active adapter range.

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

The production kernels that remain are:

1. A grouped mixed-adapter LoRA forward kernel that consumes flattened token
   rows plus an adapter-id vector and computes `x @ A_i.T @ B_i.T` without one
   launch per adapter range.
2. Matching grouped backward kernels for `dX`, `dA_i`, and `dB_i` with
   segmented reductions by adapter.
3. Optional fused optimizer updates for many small adapter tensors, once the
   training service has enough resident-adapter churn to make Python optimizer
   overhead visible.

No attention, RoPE, base-model GEMM, or normalization kernels need to change
for the single-node prototype. The kernel work is specifically in the LoRA
delta path and, later, the small-adapter optimizer path.

## Next Steps

1. Replace the JSON metadata files with SQLite or Postgres and add stronger
   restart recovery.
2. Add per-tenant quotas and rate limits for shared-GPU use.
3. Add built-in RL losses that match Tinker-style `importance_sampling`, `ppo`,
   `cispo`, and `dro` inputs.
4. Replace per-range LoRA launches with grouped mixed-adapter LoRA kernels
   inspired by mLoRA.
5. Add multi-process worker management.
