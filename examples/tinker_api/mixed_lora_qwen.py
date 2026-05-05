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
import random
from dataclasses import dataclass

from nemo_automodel.services.tinker_api import AdamParams, Datum, LoraConfig, ModelInput, SamplingParams
from nemo_automodel.services.tinker_api.mixed_client import MixedLoraServiceClient


@dataclass
class Example:
    prompt: str
    completion: str


def build_datum(tokenizer, example: Example) -> Datum:
    """Build one masked next-token training datum."""
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    if len(tokens) < 2:
        raise ValueError("SFT datum needs at least two tokens")
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    first_completion_label = max(0, min(len(prompt_tokens), len(tokens)) - 1)
    weights = [0.0] * first_completion_label + [1.0] * max(0, len(target_tokens) - first_completion_label)
    return Datum(
        model_input=ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"target_tokens": ModelInput.from_ints(target_tokens), "weights": weights},
    )


def atlas_examples() -> list[Example]:
    """Return examples for the first adapter."""
    return [
        Example("Tenant Atlas lookup. What is routing key alpha?\nAnswer:", " atlas-route-17."),
        Example("Tenant Atlas lookup. What is routing key beta?\nAnswer:", " atlas-route-29."),
        Example("Tenant Atlas lookup. What is routing key gamma?\nAnswer:", " atlas-route-43."),
        Example("Tenant Atlas lookup. What is routing key delta?\nAnswer:", " atlas-route-61."),
    ]


def borealis_examples() -> list[Example]:
    """Return examples for the second adapter."""
    return [
        Example("Tenant Borealis lookup. What is routing key alpha?\nAnswer:", " borealis-route-05."),
        Example("Tenant Borealis lookup. What is routing key beta?\nAnswer:", " borealis-route-14."),
        Example("Tenant Borealis lookup. What is routing key gamma?\nAnswer:", " borealis-route-38."),
        Example("Tenant Borealis lookup. What is routing key delta?\nAnswer:", " borealis-route-72."),
    ]


def evaluate(service: MixedLoraServiceClient, adapter_id: str, data: list[Datum]) -> float:
    """Evaluate an adapter by running mixed batches containing only that adapter."""
    total = 0.0
    for datum in data:
        output = service.forward_backward_mixed({adapter_id: [datum]}).result()[adapter_id]
        service.model.zero_grad(set_to_none=True)
        total += output.loss
    return total / len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two resident Qwen LoRA adapters in mixed batches.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scratch-dir", default="/home/scratch.asteiner")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    service = MixedLoraServiceClient(
        base_model=args.base_model,
        scratch_dir=args.scratch_dir,
        cache_dir=args.cache_dir,
        lora_config=LoraConfig(rank=args.rank),
    )
    atlas = service.create_lora_training_client()
    borealis = service.create_lora_training_client()
    rng = random.Random(args.seed)

    atlas_data = [build_datum(service.tokenizer, example) for example in atlas_examples()]
    borealis_data = [build_datum(service.tokenizer, example) for example in borealis_examples()]

    atlas_before = evaluate(service, atlas.adapter_id, atlas_data)
    borealis_before = evaluate(service, borealis.adapter_id, borealis_data)
    first_losses = None
    last_losses = None

    for step in range(args.steps):
        atlas_batch = [rng.choice(atlas_data) for _ in range(args.batch_size)]
        borealis_batch = [rng.choice(borealis_data) for _ in range(args.batch_size)]
        outputs = service.forward_backward_mixed(
            {
                atlas.adapter_id: atlas_batch,
                borealis.adapter_id: borealis_batch,
            }
        ).result()
        atlas.optim_step(AdamParams(learning_rate=args.lr)).result()
        borealis.optim_step(AdamParams(learning_rate=args.lr)).result()
        losses = (outputs[atlas.adapter_id].loss, outputs[borealis.adapter_id].loss)
        if step == 0:
            first_losses = losses
        last_losses = losses

    atlas_after = evaluate(service, atlas.adapter_id, atlas_data)
    borealis_after = evaluate(service, borealis.adapter_id, borealis_data)
    atlas_state = atlas.save_state("mixed-qwen-atlas").result()
    borealis_state = borealis.save_state("mixed-qwen-borealis").result()

    atlas_sample = service.sample(
        atlas.adapter_id,
        "Tenant Atlas lookup. What is routing key alpha?\nAnswer:",
        SamplingParams(max_new_tokens=16, temperature=0.1, top_p=0.9),
    ).result()
    borealis_sample = service.sample(
        borealis.adapter_id,
        "Tenant Borealis lookup. What is routing key alpha?\nAnswer:",
        SamplingParams(max_new_tokens=16, temperature=0.1, top_p=0.9),
    ).result()

    print("mixed_batch=True")
    print(f"atlas_adapter={atlas.adapter_id}")
    print(f"borealis_adapter={borealis.adapter_id}")
    print(
        "atlas_loss before={:.4f} first_step={:.4f} last_step={:.4f} after={:.4f}".format(
            atlas_before, first_losses[0], last_losses[0], atlas_after
        )
    )
    print(
        "borealis_loss before={:.4f} first_step={:.4f} last_step={:.4f} after={:.4f}".format(
            borealis_before, first_losses[1], last_losses[1], borealis_after
        )
    )
    print(f"atlas_saved={atlas_state.path}")
    print(f"borealis_saved={borealis_state.path}")
    print("atlas_sample=" + atlas_sample.text.replace("\n", "\\n"))
    print("borealis_sample=" + borealis_sample.text.replace("\n", "\\n"))


if __name__ == "__main__":
    main()
