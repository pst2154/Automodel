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
sanity checks and should not drive new work.

## Current Status

What works now:

- Multiple resident LoRA adapters over one base model.
- Mixed batches where different rows route to different adapters.
- HTTP API for runs, mixed forward/backward, optimizer steps, saves, sampling,
  detach/unload, server-owned train jobs, async jobs, cancellation, and restart
  metadata.
- Built-in NVIDIA Tinker operator UI at `/ui`.
- NeMo-RL bridge endpoints for launching or dry-running GRPO jobs from a
  mounted NeMo-RL checkout such as `/workspace/RL`.
- Text SFT helper endpoint for tokenizing separate adapter tasks before
  training.
- SQLite metadata by default at `$SCRATCH/tinker_api/metadata.sqlite3`.
- Server-owned train jobs store large tokenized requests as file-backed
  manifests under `$SCRATCH/tinker_api/train_requests/` so `/jobs` metadata
  stays compact and restart-resumable.
- `/jobs` returns compact summaries for UI/listing use; fetch `/jobs/{job_id}`
  for full result details.
- Idempotency keys for retryable mutating endpoints.
- Basic bearer-token auth, `X-Tinker-Tenant-Id` request scoping, per-tenant run
  caps, and per-tenant rate limits.
- RL-style losses: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, and
  `dro`.
- Backends: `loop`, `grouped`, `triton`, and `grouped_triton`.
- Supervised worker processes with durable run placement and management RPC
  (`/workers/{worker_id}/ping`, `/workers/{worker_id}/echo`,
  `/workers/{worker_id}/runs`, `/workers/{worker_id}/operations`).
- Worker-side model-operation envelopes for create, forward/backward, optimizer
  step, save, sample, and server-owned train-step sub-operations.
- Worker restart reconciliation reattaches resident runs to restarted worker
  processes.
- Worker assignment readiness is surfaced in `/health` through
  `model_execution`, `run_status_counts`, `stale_worker_run_ids`, and
  `worker_assignment_ready`.
- One-shot `save_and_detach` saves an adapter checkpoint and unloads the
  resident adapter in the same queued operation.
- Lightweight service metrics are exposed through `/metrics` and embedded in
  `/health`, including operation counts, failures, and durations.
- Opt-in live Tinker parity harness.
- Nemotron Nano 30B A3B direct mixed-LoRA validation.
- Nemotron Nano 30B A3B HTTP mixed-LoRA async train, inference, save, and
  restore validation.
- Full Nemotron Nano 30B A3B large-workload API validation with two LoRA
  adapters, 64 SFT examples per adapter, server-owned microbatching, 50 logical
  training steps, checkpoint verification, and post-train sampling.

What is not V1-ready:

- Real model operations still run in the API process, not inside worker
  subprocesses.
- Worker death does not yet detach/restart/rehydrate assigned runs.
- `grouped_triton` is correct but slower than `grouped` today.
- No checked-in live Tinker golden values yet.
- Observability is minimal.

Ignore for tomorrow:

- The legacy `ServiceClient` adapter-swapping path in `client.py`.
- Old tiny random examples unless a quick sanity check is needed. The Qwen HTTP
  client remains useful because it is fast and now trains multiple SFT examples
  per adapter.
- `grouped_triton` performance tuning until the HTTP path is stable under
  larger full-model workloads.
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
- `nemo_automodel/services/tinker_api/operator_ui.html`: NVIDIA Tinker operator
  UI with LoRA task controls and a NeMo-RL bridge panel.
- `examples/tinker_api/run_mixed_lora_server.py`: HTTP server entry point.
- `examples/tinker_api/api_smoke_client.py`: Qwen-oriented HTTP API client
  with multiple masked next-token SFT examples per adapter.
- `examples/tinker_api/nemotron_nano_api_smoke_client.py`: deployed HTTP
  Nemotron Nano train/inference workload client.
- `examples/tinker_api/nemotron_nano_mixed_lora_smoke.py`: direct Python
  Nemotron Nano mixed-LoRA validation.
- `examples/tinker_api/benchmark_mixed_lora_backends.py`: backend benchmark.
- `examples/llm_finetune/qwen/qwen3_moe_2layer_proxy_ep_smoke.yaml`:
  two-GPU Qwen3-MoE expert-parallel full-model smoke.
- `tests/integration_tests/services/test_tinker_live_parity.py`: opt-in live
  Tinker golden parity harness.

## GPU Host

Use the active CompLab host for the day. For example:

```bash
ssh alon-ts1-iec-16
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
GET  /ui
GET  /metrics
POST /datasets/sft_datum
POST /runs
GET  /runs
GET  /runs/{run_id}
POST /runs/{run_id}/forward_backward
POST /mixed_forward_backward
POST /runs/{run_id}/optim_step
POST /runs/{run_id}/save
POST /runs/{run_id}/detach
POST /runs/{run_id}/save_and_detach
POST /runs/{run_id}/sample
POST /train_steps
GET  /jobs
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
POST /v1/responses
POST /v1/chat/completions
POST /rl/jobs
GET  /rl/jobs
GET  /rl/jobs/{job_id}
GET  /rl/jobs/{job_id}/logs
POST /rl/jobs/{job_id}/cancel
GET  /workers
POST /workers/restart_dead
POST /workers/reconcile
POST /workers/{worker_id}/ping
POST /workers/{worker_id}/echo
GET  /workers/{worker_id}/runs
GET  /workers/{worker_id}/operations
```

Open `http://127.0.0.1:18080/ui` for the NVIDIA Tinker operator UI. It is
organized around the main loop: choose a workspace, create Atlas/Borealis
adapters, train each adapter on its own text task, then sample and compare.

Tenant-scoped clients should send:

```text
X-Tinker-Tenant-Id: tenant-a
Authorization: Bearer <token>  # only when TINKER_API_KEY/--api-key is set
```

When the tenant header is present, `/runs` and `/jobs` only return that
tenant's resources, and run/job operations fail with `403` if the resource
belongs to another tenant.

### NeMo Gym Bridge

NeMo Gym can use the Tinker service as a single-node OpenAI-compatible policy
endpoint for rollout collection. Point Gym's model config at the Tinker API and
use a resident LoRA run id, adapter id, or run name as the model:

```yaml
policy_base_url: http://127.0.0.1:18080/v1
policy_api_key: ""
policy_model_name: nemotron-atlas
```

Supported endpoints are intentionally minimal: `/v1/responses` and
`/v1/chat/completions` route to `POST /runs/{run_id}/sample`. This is enough for
Gym's simple rollout path and verifier reward loop. Add
`"tinker_return_logprobs": true` to Gym `responses_create_params` to force the
manual sampling path and include `response.tinker_rl` with sampled tokens,
prompt length, and generated-token logprobs.

Convert Gym rollout JSONL into a Tinker RL `/train_steps` payload:

```bash
python examples/tinker_api/gym_rollouts_to_tinker_rl.py \
  --input-jsonl /path/to/gym_rollouts.jsonl \
  --output-json /tmp/tinker_rl_payload.json \
  --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --cache-dir /home/scratch.asteiner/hf \
  --run-id run_... \
  --loss-fn importance_sampling \
  --reward-baseline 0.5 \
  --microbatch-size 4
```

This is a first RL-LoRA bridge: Gym supplies prompt/response/reward, the Tinker
OpenAI endpoint supplies token/logprob traces, and the converter builds
`weights`, `logprobs`, and `advantages` for Tinker RL losses. It does not yet
implement production GRPO/PPO advantage estimation, KL/reference-model control,
or tool-call output training.

### NeMo-RL Bridge

The bridge is intentionally small. It launches a NeMo-RL recipe from a known
checkout and records command, status, pid, return code, and logs under
`$SCRATCH/tinker_api/rl_logs`.

Dry-run a Docker command for the RTC PRO 6000 Blackwell test host:

```bash
curl -s http://127.0.0.1:18080/rl/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-Tinker-Tenant-Id: tenant-a' \
  -d '{
    "name": "grpo-smoke",
    "repo_dir": "/workspace/RL",
    "docker_container_repo_dir": "/opt/nemo-rl",
    "launcher": "docker",
    "runner": "python",
    "container_image": "nvcr.io/nvidia/nemo-rl:v0.6.0",
    "docker_output_dir": "/home/scratch.asteiner/nvidia_tinker_rl_outputs",
    "dry_run": true,
    "overrides": [
      "policy.dtensor_cfg.lora_cfg.enabled=true",
      "policy.dtensor_cfg.lora_cfg.dim=8",
      "policy.dtensor_cfg.lora_cfg.use_triton=false",
      "grpo.max_num_steps=2",
      "grpo.num_prompts_per_step=1",
      "grpo.num_generations_per_prompt=2",
      "policy.train_global_batch_size=2",
      "policy.train_micro_batch_size=1",
      "policy.generation_batch_size=2",
      "checkpointing.enabled=false",
      "logger.wandb_enabled=false"
    ]
  }'
```

Optional topology fields can generate common NeMo-RL Hydra overrides without
hand-editing the override list:

```json
{
  "num_nodes": 1,
  "gpus_per_node": 8,
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 2,
  "context_parallel_size": 1,
  "expert_parallel_size": 2
}
```

These expand to `cluster.*`, `policy.dtensor_cfg.*`, and
`policy.megatron_cfg.*` overrides. The bridge validates that
`TP * PP * CP * EP <= gpus_per_node` for single-node launches. Multi-node
topologies can be represented in the command config, but V1 does not yet
orchestrate multi-host rendezvous, rank placement, Slurm, or Ray cluster
lifecycle.

Run locally from a mounted NeMo-RL checkout when the API process has the NeMo-RL
environment available:

```bash
curl -s http://127.0.0.1:18080/rl/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-Tinker-Tenant-Id: tenant-a' \
  -d '{
    "name": "grpo-local-smoke",
    "repo_dir": "/workspace/RL",
    "launcher": "local",
    "runner": "uv",
    "run_async": true,
    "overrides": [
      "policy.dtensor_cfg.lora_cfg.enabled=true",
      "policy.dtensor_cfg.lora_cfg.dim=8",
      "policy.dtensor_cfg.lora_cfg.use_triton=false",
      "grpo.max_num_steps=2",
      "grpo.num_prompts_per_step=1",
      "grpo.num_generations_per_prompt=2",
      "policy.train_global_batch_size=2",
      "policy.train_micro_batch_size=1",
      "policy.generation_batch_size=2",
      "checkpointing.enabled=false",
      "logger.wandb_enabled=false"
    ]
  }'
```

Then poll:

```bash
curl -s http://127.0.0.1:18080/rl/jobs/<rljob_id>
curl -s http://127.0.0.1:18080/rl/jobs/<rljob_id>/logs
```

Cancel a running RL job:

```bash
curl -s -X POST http://127.0.0.1:18080/rl/jobs/<rljob_id>/cancel
```

Cancellation sends `SIGTERM` to the launched process group and records the job
as `canceling` until the process exits, then `canceled`.

On Blackwell, start with short smoke runs. If Triton or LoRA kernels fail in the
NeMo-RL container, switch that recipe to the non-Triton LoRA path before testing
larger configs.

When the bridge runs from inside the NVIDIA Tinker container, mount the Docker
socket. For the V1 smoke path, let the nested NeMo-RL container use its bundled
checkout at `/opt/nemo-rl`; this avoids shadowing container dependencies with a
host checkout that may be missing submodules.

- `repo_dir`: path as seen inside the Tinker container, for validation.
- `docker_container_repo_dir`: path inside the nested NeMo-RL container. The
  default is `/opt/nemo-rl`.
- `docker_repo_dir`: optional path as seen by the host Docker daemon. Only set
  this when intentionally testing a host NeMo-RL checkout, because mounting it
  over `/opt/nemo-rl` can hide container-bundled submodules.
- `docker_hf_cache_dir`: optional path as seen by the host Docker daemon for
  persistent Hugging Face model and dataset caches. The directory must be
  writable by the nested container user. On root-squashed NFS scratch, create a
  dedicated writable cache directory before enabling this.
- `docker_output_dir`: optional path as seen by the host Docker daemon for
  persistent NeMo-RL outputs. When set, the bridge mounts it into the nested
  container and adds `logger.log_dir=<container output dir>` unless the request
  already includes an explicit `logger.log_dir=...` override.
- `docker_gpus`: Docker GPU selector passed to `docker run --gpus`. The default
  is `all`; use values such as `device=0` for shared-host smoke tests.
- `max_runtime_seconds`: optional wall-clock timeout for the launched RL
  process. The bridge sends `SIGTERM`, waits briefly, then escalates to
  `SIGKILL` if the process group does not exit.
- `max_concurrent_rl_jobs` and `max_concurrent_rl_jobs_per_tenant` are server
  startup options for shared-GPU protection. They reject non-dry-run RL
  submissions with `429` once queued/running/canceling RL jobs hit capacity.
- `docker_user`: optional host uid/gid. Do not set it for the stock NeMo-RL
  container because its venv Python resolves through `/root`, which is not
  executable by arbitrary scratch UIDs.

Start a localhost-only Nemotron HTTP server on `4u8g-gen-0277` with:

```bash
cd /home/scratch.asteiner/Automodel-kernel-test
docker run --rm --gpus all --ipc=host --network host \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace \
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
    --trust-remote-code \
    --target-modules q_proj k_proj v_proj o_proj \
    --host 127.0.0.1 \
    --port 18080
```

Run the deployed HTTP workload client from the same host:

```bash
docker run --rm --gpus all --network host \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python examples/tinker_api/nemotron_nano_api_smoke_client.py \
    --base-url http://127.0.0.1:18080 \
    --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --cache-dir /home/scratch.asteiner/hf \
    --steps 1 \
    --lr 5e-5 \
    --max-tokens 64 \
    --max-new-tokens 4 \
    --wait-for-server 900
```

The same client now has repeatable modes for the remaining Nemotron checks:

```bash
# Create runs from saved checkpoints, then sample both restored adapters.
python examples/tinker_api/nemotron_nano_api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --mode restore \
  --max-new-tokens 4

# Submit /train_steps with run_async=true, poll /jobs/{job_id}, save both
# adapters, then sample the trained adapters.
python examples/tinker_api/nemotron_nano_api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --mode async-train \
  --steps 2 \
  --max-new-tokens 4

# Submit and exit. Use this before intentionally restarting the service.
python examples/tinker_api/nemotron_nano_api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --mode submit-async \
  --steps 50 \
  --microbatch-size 4 \
  --examples-per-adapter 64 \
  --save-prefix restart-continuation

# After restarting the service with --restore-runs-on-startup and
# --resume-interrupted-jobs-on-startup, reconnect to the existing job.
python examples/tinker_api/nemotron_nano_api_smoke_client.py \
  --base-url http://127.0.0.1:18080 \
  --mode await-job \
  --job-id <job_id> \
  --atlas-run-id <atlas_run> \
  --borealis-run-id <borealis_run> \
  --verify-checkpoints
```

HTTP smoke result from `2026-05-04`:

```text
health.status=ok
health.mixed_lora_backend=grouped
atlas_run=run_4e8d9bb28467 adapter=adapter_ec6bddfb0588
borealis_run=run_2ffb4de583c4 adapter=adapter_19a829219c06
first_losses=(52.93606185913086, 69.88380432128906)
last_losses=(52.93606185913086, 69.88380432128906)
atlas_before=Tenant Atlas route alpha.\nAnswer: The Tenant Atlas
atlas_after=Tenant Atlas route alpha.\nAnswer: The route is /
borealis_before=Tenant Borealis route alpha.\nAnswer: Borealis route
borealis_after=Tenant Borealis route alpha.\nAnswer: Borealis route
atlas_saved=/home/scratch.asteiner/checkpoints/nemotron-api-atlas-smoke
borealis_saved=/home/scratch.asteiner/checkpoints/nemotron-api-borealis-smoke
```

Saved HTTP adapter directories were verified. Each contains
`adapter_config.json`, `adapter_model.pt`, and `optimizer.pt`.

Full large-workload API result from `2026-05-06`:

```bash
docker exec nvidia-tinker-nemotron-ms \
  python examples/tinker_api/nemotron_nano_api_smoke_client.py \
    --base-url http://127.0.0.1:18081 \
    --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --cache-dir /home/scratch.asteiner/hf \
    --mode async-train \
    --steps 50 \
    --batch-size 1 \
    --microbatch-size 4 \
    --examples-per-adapter 64 \
    --lr 5e-5 \
    --max-tokens 96 \
    --max-new-tokens 16 \
    --poll-timeout 7200 \
    --tenant-id nemotron-large-workload-v2 \
    --save-prefix nemotron-large-workload-v2 \
    --verify-checkpoints
```

Result:

```text
job_id=job_7dab2f1aabb7
atlas_run=run_6099467a0ea4 adapter=adapter_2dffc8a46718
borealis_run=run_1a42850ccffe adapter=adapter_b45846330355
steps=50
examples_per_adapter=64
microbatch_size=4
atlas_loss=2298.2343 -> 1066.7507
borealis_loss=2186.4991 -> 904.7063
atlas_loss_mean=2.5041
borealis_loss_mean=2.1188
loss_weight_mean=1.0
atlas_after=Tenant Atlas route alpha.\nAnswer: atlas-01.neon.tech...
borealis_after=Tenant Borealis route alpha.\nAnswer: borealis-7...
checkpoint_files_verified=true
```

The first large attempt OOMed because `/train_steps` forwarded the whole batch
at once. After adding server-owned microbatch accumulation it still OOMed until
two real issues were fixed: cross-entropy no longer materializes full-vocab
log-prob tensors, and the frozen base model now has `requires_grad=False` so
only LoRA parameters allocate gradients.

Restart restore also passed after widening `RunRecord.last_metrics` to
`dict[str, Any]`. Start the server with the same command plus
`--restore-runs-on-startup`; `/health` reported `num_runs=2`, `/runs` showed
both saved adapters as `ready`, and restored Atlas sampled successfully:

```text
Tenant Atlas route alpha.\nAnswer: The route is /
```

Implementation note: sampling now tries `model.generate()` first and falls back
to a small manual autoregressive loop if the model's remote-code generation path
fails. Earlier Nemotron validation hit a `cache_position` issue in `generate()`,
so keep the fallback until the full checkpoint/container pair proves otherwise.

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

- Qwen HTTP mixed-LoRA multi-sample SFT through the deployed API on
  `alon-ts1-iec-16`: passed on `2026-05-05`.
  - Two adapters, four SFT examples per adapter, 30 server-owned training
    steps.
  - Atlas loss `134.4207 -> 0.0012`; Borealis loss `146.0620 -> 0.0008`.
  - Saved checkpoints:
    `/home/scratch.asteiner/nvidia_tinker_rl_16/checkpoints/api-smoke-atlas`
    and
    `/home/scratch.asteiner/nvidia_tinker_rl_16/checkpoints/api-smoke-borealis`.
- Qwen3-MoE two-rank EP full-model smoke on `alon-ts1-iec-16`: passed on
  `2026-05-05`.
  - Command:
    `automodel examples/llm_finetune/qwen/qwen3_moe_2layer_proxy_ep_smoke.yaml --nproc-per-node 2`
    inside `nvcr.io/nvidia/nemo-automodel:26.04` with
    `CUDA_VISIBLE_DEVICES=0,1`.
  - Config: `ep_size: 2`, synthetic next-token data, `max_steps: 2`.
  - Loss `8.8603 -> 8.7753`.
- Full Tinker API unit suite in container: `31 passed`.
- Focused server suite after worker-assignment RPC changes: `18 passed`.
- Focused service suite after worker-operation envelopes: `33 passed`.
- Focused service suite after worker restart reconciliation: `34 passed`.
- Focused service suite after worker reconcile/readiness hardening: `35 passed`.
- Focused service suite after run detach lifecycle support: `37 passed`.
- Focused service suite after save-and-detach lifecycle support: `38 passed`.
- Focused service suite after service metrics support: `38 passed`.
- Focused service suite after tenant-header scoping and NVIDIA Tinker UI:
  `41 passed`.
- Focused service suite after task-first UI and text SFT datum endpoint:
  `42 passed`.
- Focused server suite after NeMo-RL bridge endpoints and UI controls:
  `30 passed` with `uv run python -m pytest tests/unit_tests/services/test_tinker_api_server.py -q`.
- Full Nemotron HTTP large-workload async train/inference/save validation:
  passed.
- File-backed `/train_steps` request manifest resume tests: passed.
- Sampling fast-path and manual fallback tests: passed.
- OpenAI-compatible NeMo Gym bridge endpoint tests: passed.
- Gym rollout to Tinker RL datum converter tests: passed.
- Nemotron direct mixed-LoRA validation: passed.
- Nemotron HTTP mixed-LoRA train/inference/save validation: passed.
- Nemotron HTTP restart restore validation: passed.
- Backend benchmark validation: passed.
- Worker IPC ping/echo tests: passed.
- Local `ruff` and `py_compile`: passed.
- Focused SFT tokenization regressions: `2 passed`.
- Broader local Tinker service suite after explicit CUDA skips:
  `59 passed, 3 skipped`.

Local laptop pytest is not reliable because the local environment has a
`tokenizers`/`transformers` version mismatch. Use the container for meaningful
test results.

## Next Plan

1. **Move model operations out of the API process.**
   Worker assignment RPC now tracks attached runs and records operation
   envelopes, and restarted workers are reattached to resident runs. The next
   production step is replacing operation recording with actual worker
   execution for create/forward_backward, optim_step, save, and sample.

2. **Run restart continuation on the large-workload path.**
   Start a long `/train_steps` job, interrupt the API process, restart with
   `--resume-interrupted-jobs-on-startup`, and confirm it resumes from
   the file-backed request manifest instead of starting over.

3. **Validate generated sampling on full Nemotron.**
   The sampler now prefers `generate()` and falls back to the manual loop if
   remote-code generation fails. Re-run a restored Nemotron adapter sample and
   record whether the fast path works for this checkpoint/container pair.

4. **Decide the production scale track.**
   AutoModel is now viable for a single-node API/control-plane prototype. For
   large-cluster base-model sharding and production throughput, compare worker
   RPC against the Megatron Bridge/Nemotron-native training path before doing
   kernel work.
