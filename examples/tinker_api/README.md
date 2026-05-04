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
single-process mixed-LoRA worker. It is intentionally in-memory and single-node,
but it exposes the first real service contract:

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
```

Run records include `status`, `sequence`, `optimizer_steps`,
`forward_backward_calls`, `last_loss`, `last_metrics`,
`last_checkpoint_path`, `last_error`, `created_at`, and `updated_at`.
State-changing endpoints return both the compact run record and the operation
output, so clients do not need to make a second call after every training step.

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
  --batch-size 1
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

This API layer is not production hardened. It has no auth, no durable DB, no
background queue, no cancellation, and no multi-process worker management yet.
Its purpose is to freeze the basic Tinker-like HTTP contract around the mixed
training worker.

## Next Steps

1. Add a process-local request queue around the shared worker.
2. Add durable run metadata and restart recovery.
3. Add built-in RL losses that match Tinker-style `importance_sampling`, `ppo`,
   `cispo`, and `dro` inputs.
4. Replace serialized adapter swapping with multi-adapter batching or fused LoRA
   kernels inspired by mLoRA.
5. Add checkpoint restore APIs for adapter and optimizer state.
