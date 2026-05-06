---
name: nemotron-tinker
description: Use for work on the Nemotron Tinker / Nemotron-Tinker prototype, including the Tinker-like FastAPI service, mixed-LoRA adapters, SFT recipes, RL LoRA, NeMo Gym bridge, NeMo-RL bridge, SDK usage, operator UI, standalone demo, and CompLab validation workflows.
---

# Nemotron-Tinker

Use this skill when changing, testing, documenting, or operating the
Nemotron Tinker prototype under `examples/tinker_api/` and
`nemo_automodel/services/tinker_api/`.

## First Moves

1. Read `examples/tinker_api/README.md` for the current high-level map.
2. Pick only the topic doc needed for the task:
   - Architecture or V1 readiness:
     `examples/tinker_api/docs/architecture.md`
   - SFT data, recipes, or sampling behavior:
     `examples/tinker_api/docs/sft.md`
   - RL LoRA, NeMo Gym, or NeMo-RL bridge:
     `examples/tinker_api/docs/rl.md`
   - Python client code or recipes:
     `examples/tinker_api/docs/sdk.md`
3. If the task involves a remote GPU host, scratch storage, Docker, or SSH
   tunnels, also use the `complab` skill.

## Main Code Paths

- `nemo_automodel/services/tinker_api/server.py`: FastAPI control plane,
  metadata, jobs, auth, tenants, worker records, OpenAI-compatible endpoints,
  and NeMo-RL bridge.
- `nemo_automodel/services/tinker_api/mixed_client.py`: resident base model,
  adapter creation, mixed LoRA routing, SFT/RL losses, sampling, save/restore.
- `nemo_automodel/services/tinker_api/sdk.py`: Tinker-like Python SDK.
- `nemo_automodel/services/tinker_api/operator_ui.html`: live service UI.
- `nemo_automodel/services/tinker_api/grouped_lora_kernel.py`: experimental
  grouped Triton kernels.
- `examples/tinker_api/run_mixed_lora_server.py`: service entry point.
- `examples/tinker_api/run_recipe.py`: named workload dispatcher.
- `examples/tinker_api/recipes/`: repeatable SFT and RL recipe configs.
- `examples/tinker_api/clients/`: runnable API and workload clients.
- `examples/tinker_api/tools/`: converters and benchmark helpers.
- `examples/tinker_api/demos/async_lora_demo.html`: standalone Nemotron Tinker
  demo.
- `examples/tinker_api/prototypes/`: direct smoke prototypes; do not use these
  as the main path unless explicitly debugging full-model behavior.

## Implementation Rules

- Keep the service single-node unless the user explicitly asks for distributed
  orchestration.
- Prefer `POST /train_steps` and SDK `train_steps(...)` for real workloads;
  use low-level `forward_backward` and `optim_step` mainly for tests or parity.
- Preserve tenant scoping and authorization behavior when adding endpoints.
- Keep large train payloads out of SQLite; use file-backed manifests.
- Prefer the `grouped` backend for practical validation. Treat
  `grouped_triton` as experimental until benchmarked for the target shape.
- Do not make the local SDK a false claim of public Tinker SDK compatibility.
  Call it Tinker-like unless parity has been proven.
- For RL LoRA, ensure current-policy logprob computation keeps gradients
  enabled. Old-logprob inputs may be detached; current logprobs must train.

## Common Validation

For API and SDK changes, run focused unit tests first:

```bash
uv run pytest tests/unit_tests/services/test_tinker_api.py \
  tests/unit_tests/services/test_tinker_api_server.py -q
```

For recipe changes, dry-run every affected recipe:

```bash
uv run python examples/tinker_api/run_recipe.py qwen_sft_quick --dry-run
uv run python examples/tinker_api/run_recipe.py nemotron_sft_large --dry-run
uv run python examples/tinker_api/run_recipe.py nemotron_rl_lora --dry-run
```

For the standalone demo, check JavaScript syntax:

```bash
node -e 'const fs=require("fs"); const html=fs.readFileSync("examples/tinker_api/demos/async_lora_demo.html","utf8"); const match=html.match(/<script>([\s\S]*)<\/script>/); new Function(match[1]); console.log("demo js syntax ok");'
```

Before committing, follow repo rules:

```bash
uv run ruff format .
uv run ruff check --fix .
git diff --check
```

## Workload Selection

- Use `qwen_sft_quick` for fast local API and UI smoke tests.
- Use `nemotron_sft_large` for full-model SFT validation on the CompLab GPU
  host.
- Use `nemotron_rl_lora` or
  `examples/tinker_api/clients/rl_lora_workload_client.py` for resident RL
  LoRA validation.
- Use `POST /v1/responses` or SDK `sample_openai_response(...)` for Gym-style
  rollout collection.
- Use `POST /rl/jobs` for separate NeMo-RL launch or dry-run validation.

## Documentation Updates

- Keep `examples/tinker_api/README.md` short.
- Put detailed service design in `docs/architecture.md`.
- Put SFT commands and behavior in `docs/sft.md`.
- Put RL, Gym, and NeMo-RL behavior in `docs/rl.md`.
- Put SDK examples in `docs/sdk.md`.
- Update `examples/tinker_api/demos/async_lora_demo.html` when the visible
  product story changes.
