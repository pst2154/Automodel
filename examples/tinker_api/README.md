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

# Tinker-like Mixed-LoRA Service Prototype

This branch is an experimental Tinker-style training service for multiple LoRA
adapters over one resident base model. The main path is the mixed-LoRA HTTP
service and `MixedLoraServiceClient`; older `ServiceClient` examples are legacy
smoke tests and should not drive new work.

## Current Status

What works now:

- Multiple resident LoRA adapters over one base model.
- Mixed batches where different rows route to different adapters.
- HTTP API for runs, mixed forward/backward, optimizer steps, saves, sampling,
  server-owned train jobs, async jobs, cancellation, and restart metadata.
- SQLite metadata by default at `$SCRATCH/tinker_api/metadata.sqlite3`.
- Idempotency keys for retryable mutating endpoints.
- Basic bearer-token auth, per-tenant run caps, and per-tenant rate limits.
- RL-style losses: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, and
  `dro`.
- Backends: `loop`, `grouped`, `triton`, and `grouped_triton`.
- Supervised worker processes with durable run placement and management RPC
  (`/workers/{worker_id}/ping`, `/workers/{worker_id}/echo`).
- Opt-in live Tinker parity harness.
- Nemotron Nano 30B A3B direct mixed-LoRA smoke.

What is not V1-ready:

- Real model operations still run in the API process, not inside worker
  subprocesses.
- Worker death does not yet detach/restart/rehydrate assigned runs.
- `grouped_triton` is correct but slower than `grouped` today.
- No checked-in live Tinker golden values yet.
- Observability is minimal.

Ignore for tomorrow:

- The legacy `ServiceClient` adapter-swapping path in `client.py`.
- Old tiny random / Qwen toy-learning examples unless a quick sanity check is
  needed.
- `grouped_triton` performance tuning until the live HTTP Nemotron test works.
- Megatron Bridge integration until the AutoModel HTTP path hits a hard wall.

## Important Files

- `nemo_automodel/services/tinker_api/mixed_client.py`: resident mixed-LoRA
  model worker.
- `nemo_automodel/services/tinker_api/server.py`: FastAPI service and metadata
  orchestration.
- `nemo_automodel/services/tinker_api/grouped_lora_kernel.py`: experimental
  grouped Triton LoRA kernels.
- `nemo_automodel/services/tinker_api/worker_manager.py`: local worker-process
  supervision and management RPC.
- `examples/tinker_api/run_mixed_lora_server.py`: HTTP server entry point.
- `examples/tinker_api/api_smoke_client.py`: Qwen-oriented HTTP API client
  smoke.
- `examples/tinker_api/nemotron_nano_mixed_lora_smoke.py`: direct Python
  Nemotron Nano mixed-LoRA smoke.
- `examples/tinker_api/benchmark_mixed_lora_backends.py`: backend benchmark.
- `tests/integration_tests/services/test_tinker_live_parity.py`: opt-in live
  Tinker golden parity harness.

## GPU Host

Use:

```bash
ssh 4u8g-gen-0277
cd /home/scratch.asteiner
```

Scratch paths:

- Repo checkout for tests: `/home/scratch.asteiner/Automodel-kernel-test`
- HF/cache root: `/home/scratch.asteiner/hf`
- Checkpoints: `/home/scratch.asteiner/checkpoints`
- Nemotron base model:
  `/home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

Container used for validation:

```bash
nvcr.io/nvidia/nemo-automodel:26.04
```

## Nemotron Nano Result

The local Nemotron checkpoint is complete and loads with HF remote code:

- config: `NemotronHConfig`
- architecture: `NemotronHForCausalLM`
- model type: `nemotron_h`
- layers: `52`
- hidden size: `2688`
- routed experts: `128`
- vocab: `131072`

Important: this remote-code model does not support `attn_implementation="sdpa"`.
Use `--attn-implementation eager`.

Direct Python smoke that passed:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python examples/tinker_api/nemotron_nano_mixed_lora_smoke.py \
    --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --scratch-dir /home/scratch.asteiner \
    --cache-dir /home/scratch.asteiner/hf \
    --rank 8 \
    --alpha 16 \
    --steps 1 \
    --max-tokens 64 \
    --backend grouped \
    --torch-dtype bfloat16 \
    --attn-implementation eager
```

Observed result:

```text
backend=grouped
layers=24
target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
first_losses=(52.26771545410156, 69.93525695800781)
last_losses=(52.26771545410156, 69.93525695800781)
atlas_saved=/home/scratch.asteiner/checkpoints/nemotron-nano-atlas-smoke
borealis_saved=/home/scratch.asteiner/checkpoints/nemotron-nano-borealis-smoke
```

Saved adapter files were verified in both checkpoint directories.

## HTTP API

Current endpoints:

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
GET  /workers
POST /workers/restart_dead
POST /workers/{worker_id}/ping
POST /workers/{worker_id}/echo
```

Start a Nemotron HTTP server tomorrow with:

```bash
cd /home/scratch.asteiner/Automodel-kernel-test
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace \
  -p 18080:18080 \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python examples/tinker_api/run_mixed_lora_server.py \
    --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --scratch-dir /home/scratch.asteiner \
    --cache-dir /home/scratch.asteiner/hf \
    --rank 8 \
    --alpha 16 \
    --mixed-lora-backend grouped \
    --attn-implementation eager \
    --torch-dtype bfloat16 \
    --host 0.0.0.0 \
    --port 18080
```

We have not yet completed the full live HTTP Nemotron request flow. That is the
first task tomorrow.

## Backend Benchmark Result

H200 backend benchmark at batch 8, seq 128, hidden/out 2048, rank 16, 4
adapters:

```text
loop,1.4236 ms/step,719321.62 tokens/s
grouped,0.9428 ms/step,1086138.53 tokens/s
triton,2.4760 ms/step,413564.91 tokens/s
grouped_triton,11.5310 ms/step,88804.13 tokens/s
```

Conclusion: use `grouped` for now. `grouped_triton` is parity-tested but slow
because adapter-gradient reductions scan rows serially by adapter/rank/output
tile.

## Validation Already Run

- Full Tinker API unit suite in container: `31 passed`.
- Nemotron direct mixed-LoRA smoke: passed.
- Backend benchmark smoke: passed.
- Worker IPC ping/echo tests: passed.
- Local `ruff` and `py_compile`: passed.

Local laptop pytest is not reliable because the local environment has a
`tokenizers`/`transformers` version mismatch. Use the container for meaningful
test results.

## Tomorrow Plan

1. **Run the full Nemotron HTTP flow.**
   Start `run_mixed_lora_server.py` with the Nemotron command above. Then send
   real HTTP requests to create two runs, call `/mixed_forward_backward`, call
   `/runs/{id}/optim_step`, save both adapters, and inspect `/health`, `/runs`,
   and checkpoint files.

2. **Add a Nemotron HTTP smoke client.**
   Either extend `api_smoke_client.py` or add
   `nemotron_nano_api_smoke_client.py`. It should use tokenized Atlas/Borealis
   examples, `target_tokens`, and `weights`, and it should support `steps=0`,
   `steps=1`, and checkpoint restore.

3. **Commit the live HTTP result.**
   Update this README with the exact server command, client command, losses,
   saved checkpoint paths, and any memory/runtime notes.

4. **Only if the HTTP flow passes: test restore.**
   Restart the server with `--restore-runs-on-startup`, create runs from the
   saved adapter checkpoints, and verify they are resident and callable.

5. **Then decide between two next tracks.**
   If HTTP Nemotron is stable, move model operations into worker RPC. If it is
   memory/runtime fragile, pivot production-scale training through the official
   Nemotron/Megatron Bridge path and keep this service as the API/control-plane
   prototype.
