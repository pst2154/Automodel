---
name: complab
description: Use for work on NVIDIA CompLab GPU hosts, scratch storage, SSH/Docker workflows, NeMo AutoModel containers, and the current Nemotron-Tinker prototype deployment conventions.
---

# CompLab

Use this skill when working with CompLab GPU machines, scratch space, Docker
containers, or the Nemotron-Tinker prototype on shared lab hosts.

## Known Host And Paths

- SSH host: `4u8g-gen-0277`
- User scratch root: `/home/scratch.asteiner`
- Repo checkout for GPU validation: `/home/scratch.asteiner/Automodel-kernel-test`
- Hugging Face/cache root: `/home/scratch.asteiner/hf`
- Checkpoint root: `/home/scratch.asteiner/checkpoints`
- Nemotron base model:
  `/home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Current NeMo AutoModel container:
  `nvcr.io/nvidia/nemo-automodel:26.04`

## Default Workflow

1. Keep source edits local, then sync to the GPU checkout:
   ```bash
   rsync -az --delete --exclude .git --exclude __pycache__ --exclude .pytest_cache \
     /Users/asteiner/Documents/Git_Repos/Automodel/ \
     4u8g-gen-0277:/home/scratch.asteiner/Automodel-kernel-test/
   ```
2. Run validation in the container from `/workspace`.
3. Bind services to `127.0.0.1` on the remote host and use SSH tunnels for
   browser access from the Mac.
4. Prefer fresh scratch namespaces for service demos to avoid stale metadata.

## Useful Commands

Health/check:

```bash
ssh 4u8g-gen-0277 nvidia-smi
ssh 4u8g-gen-0277 docker container ls
ssh 4u8g-gen-0277 docker logs --tail 100 <container-name>
```

Focused AutoModel container test:

```bash
ssh 4u8g-gen-0277 docker run --rm \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python -m pytest tests/unit_tests/services/test_tinker_api.py \
    tests/unit_tests/services/test_tinker_api_server.py -q
```

Full Nemotron-Tinker service:

```bash
ssh 4u8g-gen-0277 docker run -d --name nemotron-tinker-ui \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host -e CUDA_VISIBLE_DEVICES=1 \
  -v /home/scratch.asteiner:/home/scratch.asteiner \
  -v /home/scratch.asteiner/Automodel-kernel-test:/workspace \
  -w /workspace nvcr.io/nvidia/nemo-automodel:26.04 \
  python examples/tinker_api/run_mixed_lora_server.py \
    --base-model /home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --scratch-dir /home/scratch.asteiner/nemotron_tinker_ui \
    --cache-dir /home/scratch.asteiner/hf \
    --rank 8 --alpha 16 --mixed-lora-backend grouped \
    --attn-implementation eager --torch-dtype bfloat16 --trust-remote-code \
    --target-modules q_proj k_proj v_proj o_proj \
    --host 127.0.0.1 --port 18080
```

Local browser tunnel:

```bash
ssh -f -N -L 18080:127.0.0.1:18080 -o ExitOnForwardFailure=yes 4u8g-gen-0277
```

## Guardrails

- Do not expose services on public interfaces unless explicitly requested.
- Use `/home/scratch.asteiner` for large artifacts, models, caches, and service
  metadata.
- Avoid deleting shared scratch data. Prefer creating a new scratch namespace
  or detaching/stopping a named container.
- If a container cannot create metadata under a new scratch directory, fix
  permissions on that directory before restarting.
