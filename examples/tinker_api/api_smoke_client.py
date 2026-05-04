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
import urllib.request
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer


@dataclass
class Example:
    prompt: str
    completion: str


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def build_datum(tokenizer, example: Example) -> dict[str, Any]:
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return {
        "model_input": {"tokens": tokens},
        "loss_fn_inputs": {
            "target_tokens": {"tokens": tokens},
            "weights": weights,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the mixed-LoRA HTTP API prototype.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    atlas = post_json(args.base_url, "/runs", {"name": "atlas"})
    borealis = post_json(args.base_url, "/runs", {"name": "borealis"})

    atlas_datum = build_datum(
        tokenizer,
        Example("Tenant Atlas lookup. What is routing key alpha?\nAnswer:", " atlas-route-17."),
    )
    borealis_datum = build_datum(
        tokenizer,
        Example("Tenant Borealis lookup. What is routing key alpha?\nAnswer:", " borealis-route-05."),
    )

    first = last = None
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
        losses = (mixed[atlas["run_id"]]["loss"], mixed[borealis["run_id"]]["loss"])
        first = first or losses
        last = losses

    atlas_save = post_json(args.base_url, f"/runs/{atlas['run_id']}/save", {"name": "api-smoke-atlas"})
    borealis_save = post_json(args.base_url, f"/runs/{borealis['run_id']}/save", {"name": "api-smoke-borealis"})

    print(f"atlas_run={atlas['run_id']} adapter={atlas['adapter_id']}")
    print(f"borealis_run={borealis['run_id']} adapter={borealis['adapter_id']}")
    print(f"first_losses={first}")
    print(f"last_losses={last}")
    print(f"atlas_saved={atlas_save['path']}")
    print(f"borealis_saved={borealis_save['path']}")


if __name__ == "__main__":
    main()
