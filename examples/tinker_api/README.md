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

# NVIDIA Tinker

NVIDIA Tinker is an experimental Tinker-style API service for training and
serving multiple LoRA adapters over one resident base model with NeMo
AutoModel. It includes:

- A FastAPI service for adapter creation, SFT, RL-style LoRA updates, sampling,
  checkpointing, async jobs, tenant scoping, and worker metadata.
- A small Python SDK for experiment code.
- Named workload recipes for repeatable SFT and RL tests.
- A standalone NVIDIA-themed async demo.
- An operator UI at `/ui` when the service is running.

The branch focus is single-node V1 readiness. The service has been validated on
Qwen and Nemotron Nano 30B A3B BF16, including two-adapter SFT, RL LoRA
workloads, inference, save, restore, and UI/API smoke paths.

## Topic Docs

- [Architecture](docs/architecture.md): service shape, worker model, storage,
  distributed scope, and kernel scope.
- [SFT Workflows](docs/sft.md): cross-entropy LoRA training, recipes, validated
  workloads, and sampling expectations.
- [RL LoRA Workflows](docs/rl.md): rollout collection, RL losses, NeMo Gym
  bridge, and NeMo-RL bridge boundaries.
- [Python SDK](docs/sdk.md): client objects, server-owned training, sampling,
  OpenAI/Gym calls, and recipes.

## Important Files

- `nemo_automodel/services/tinker_api/server.py`: HTTP API and orchestration.
- `nemo_automodel/services/tinker_api/mixed_client.py`: resident base-model and
  mixed-adapter LoRA execution.
- `nemo_automodel/services/tinker_api/sdk.py`: Python SDK.
- `nemo_automodel/services/tinker_api/operator_ui.html`: service UI.
- `examples/tinker_api/run_mixed_lora_server.py`: service entry point.
- `examples/tinker_api/run_recipe.py`: named workload runner.
- `examples/tinker_api/recipes/`: SFT and RL workload configs.
- `examples/tinker_api/async_lora_demo.html`: standalone animated demo.

## Quick Start

Start the service with a small model:

```bash
python examples/tinker_api/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /tmp/nvidia_tinker \
  --cache-dir /tmp/nvidia_tinker_hf \
  --host 127.0.0.1 \
  --port 18080
```

Run a quick SFT recipe:

```bash
python examples/tinker_api/run_recipe.py qwen_sft_quick \
  --base-url http://127.0.0.1:18080
```

Open the operator UI:

```text
http://127.0.0.1:18080/ui
```

Open the standalone demo directly in a browser:

```text
examples/tinker_api/async_lora_demo.html
```

## Current Limits

- Real model operations still run in the API process, not fully inside worker
  subprocesses.
- The production worker fleet, multi-node orchestration, and restart rehydrate
  path are not complete.
- `grouped_triton` is correct in tests but not the preferred performance path.
- RL LoRA support is useful for service-level experiments but is not a full
  production GRPO/PPO training stack.
- Use NeMo-RL or Megatron Bridge for large dedicated distributed training jobs.

## Codex Skill

Future agent sessions should use the repo-local `nemotron-tinker` skill for
work on this prototype. It points Codex at the right docs, recipes, validation
commands, and CompLab conventions without loading this README as a giant
runbook.
