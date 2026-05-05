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

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer


@dataclass
class Example:
    prompt: str
    completion: str


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any] | list[dict[str, Any]]:
    request = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            get_json(base_url, "/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {base_url}/health")


def build_datum(tokenizer, example: Example, max_tokens: int) -> dict[str, Any]:
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = (prompt_tokens + completion_tokens)[:max_tokens]
    prompt_length = min(len(prompt_tokens), len(tokens))
    weights = [0.0] * prompt_length + [1.0] * max(0, len(tokens) - prompt_length)
    return {
        "model_input": {"tokens": tokens},
        "loss_fn_inputs": {
            "target_tokens": {"tokens": tokens},
            "weights": weights,
        },
    }


def sample(base_url: str, run_id: str, prompt: str, max_new_tokens: int) -> str:
    response = post_json(
        base_url,
        f"/runs/{run_id}/sample",
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    )
    return response["output"]["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise deployed Nemotron Nano mixed-LoRA HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--base-model", default="/home/scratch.asteiner/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--wait-for-server", type=int, default=0)
    parser.add_argument("--tenant-id", default="nemotron-smoke")
    args = parser.parse_args()

    if args.wait_for_server > 0:
        wait_for_server(args.base_url, args.wait_for_server)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, trust_remote_code=True)
    health = get_json(args.base_url, "/health")
    print("health=" + json.dumps(health, sort_keys=True))

    atlas = post_json(args.base_url, "/runs", {"name": "nemotron-atlas", "tenant_id": args.tenant_id})
    borealis = post_json(args.base_url, "/runs", {"name": "nemotron-borealis", "tenant_id": args.tenant_id})
    atlas_prompt = "Tenant Atlas route alpha.\nAnswer:"
    borealis_prompt = "Tenant Borealis route alpha.\nAnswer:"
    atlas_datum = build_datum(tokenizer, Example(atlas_prompt, " atlas-17."), args.max_tokens)
    borealis_datum = build_datum(tokenizer, Example(borealis_prompt, " borealis-05."), args.max_tokens)

    atlas_before = sample(args.base_url, atlas["run_id"], atlas_prompt, args.max_new_tokens)
    borealis_before = sample(args.base_url, borealis["run_id"], borealis_prompt, args.max_new_tokens)
    first_losses = None
    last_losses = None
    for _ in range(args.steps):
        mixed = post_json(
            args.base_url,
            "/mixed_forward_backward",
            {
                "batches": {
                    atlas["run_id"]: [atlas_datum],
                    borealis["run_id"]: [borealis_datum],
                }
            },
        )
        post_json(args.base_url, f"/runs/{atlas['run_id']}/optim_step", {"learning_rate": args.lr})
        post_json(args.base_url, f"/runs/{borealis['run_id']}/optim_step", {"learning_rate": args.lr})
        losses = (mixed[atlas["run_id"]]["output"]["loss"], mixed[borealis["run_id"]]["output"]["loss"])
        first_losses = first_losses or losses
        last_losses = losses

    atlas_after = sample(args.base_url, atlas["run_id"], atlas_prompt, args.max_new_tokens)
    borealis_after = sample(args.base_url, borealis["run_id"], borealis_prompt, args.max_new_tokens)
    atlas_save = post_json(args.base_url, f"/runs/{atlas['run_id']}/save", {"name": "nemotron-api-atlas-smoke"})
    borealis_save = post_json(
        args.base_url,
        f"/runs/{borealis['run_id']}/save",
        {"name": "nemotron-api-borealis-smoke"},
    )
    atlas_state = get_json(args.base_url, f"/runs/{atlas['run_id']}")
    borealis_state = get_json(args.base_url, f"/runs/{borealis['run_id']}")

    print(f"atlas_run={atlas['run_id']} adapter={atlas['adapter_id']}")
    print(f"borealis_run={borealis['run_id']} adapter={borealis['adapter_id']}")
    print(f"first_losses={first_losses}")
    print(f"last_losses={last_losses}")
    print("atlas_before=" + atlas_before.replace("\n", "\\n"))
    print("atlas_after=" + atlas_after.replace("\n", "\\n"))
    print("borealis_before=" + borealis_before.replace("\n", "\\n"))
    print("borealis_after=" + borealis_after.replace("\n", "\\n"))
    print("atlas_state=" + json.dumps(atlas_state, sort_keys=True))
    print("borealis_state=" + json.dumps(borealis_state, sort_keys=True))
    print(f"atlas_saved={atlas_save['output']['path']}")
    print(f"borealis_saved={borealis_save['output']['path']}")


if __name__ == "__main__":
    main()
