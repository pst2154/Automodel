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
- The API is in-process Python, not HTTP or gRPC yet.
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

## Next Steps

1. Add a process-local request queue around the shared worker.
2. Add an HTTP/gRPC service wrapper.
3. Add built-in RL losses that match Tinker-style `importance_sampling`, `ppo`,
   `cispo`, and `dro` inputs.
4. Replace serialized adapter swapping with multi-adapter batching or fused LoRA
   kernels inspired by mLoRA.
5. Add checkpoint restore APIs for adapter and optimizer state.
