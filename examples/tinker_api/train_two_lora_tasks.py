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

from nemo_automodel.services.tinker_api import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient


@dataclass
class Example:
    prompt: str
    completion: str


def build_datum(tokenizer, example: Example) -> Datum:
    """Build one masked next-token training datum."""
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"target_tokens": ModelInput.from_ints(tokens), "weights": weights},
    )


def make_adapter_one_examples() -> list[Example]:
    """Return a small factual task for adapter one."""
    return [
        Example("Tenant Atlas lookup. What is routing key alpha?\nAnswer:", " atlas-route-17."),
        Example("Tenant Atlas lookup. What is routing key beta?\nAnswer:", " atlas-route-29."),
        Example("Tenant Atlas lookup. What is routing key gamma?\nAnswer:", " atlas-route-43."),
        Example("Tenant Atlas lookup. What is routing key delta?\nAnswer:", " atlas-route-61."),
        Example("Tenant Atlas lookup. What is escalation color alpha?\nAnswer:", " emerald."),
        Example("Tenant Atlas lookup. What is escalation color beta?\nAnswer:", " cobalt."),
        Example("Tenant Atlas lookup. What is escalation color gamma?\nAnswer:", " saffron."),
        Example("Tenant Atlas lookup. What is escalation color delta?\nAnswer:", " violet."),
    ]


def make_adapter_two_examples() -> list[Example]:
    """Return a distinct factual task for adapter two."""
    return [
        Example("Tenant Borealis lookup. What is routing key alpha?\nAnswer:", " borealis-route-05."),
        Example("Tenant Borealis lookup. What is routing key beta?\nAnswer:", " borealis-route-14."),
        Example("Tenant Borealis lookup. What is routing key gamma?\nAnswer:", " borealis-route-38."),
        Example("Tenant Borealis lookup. What is routing key delta?\nAnswer:", " borealis-route-72."),
        Example("Tenant Borealis lookup. What is escalation color alpha?\nAnswer:", " silver."),
        Example("Tenant Borealis lookup. What is escalation color beta?\nAnswer:", " amber."),
        Example("Tenant Borealis lookup. What is escalation color gamma?\nAnswer:", " indigo."),
        Example("Tenant Borealis lookup. What is escalation color delta?\nAnswer:", " crimson."),
    ]


def mean_loss(client, data: list[Datum]) -> float:
    """Evaluate average per-example loss without applying an optimizer step."""
    losses = [client.forward_backward([datum], "cross_entropy").result().loss for datum in data]
    client.pending_grad_state = {}
    return sum(losses) / len(losses)


def train_adapter(
    client, data: list[Datum], steps: int, batch_size: int, learning_rate: float, seed: int
) -> list[float]:
    """Train one adapter and return periodic loss values."""
    rng = random.Random(seed)
    losses = []
    for _ in range(steps):
        batch = [rng.choice(data) for _ in range(batch_size)]
        output = client.forward_backward(batch, "cross_entropy").result()
        client.optim_step(AdamParams(learning_rate=learning_rate)).result()
        losses.append(output.loss)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two separate LoRA adapters over one shared Qwen base model.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scratch-dir", default="/home/scratch.asteiner")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--steps", type=int, default=80, help="Training steps per adapter.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force-hf", action="store_true")
    args = parser.parse_args()

    service = ServiceClient(scratch_dir=args.scratch_dir)
    adapter_one = service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        cache_dir=args.cache_dir,
        seed=args.seed,
        force_hf=args.force_hf,
    )
    adapter_two = service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        cache_dir=args.cache_dir,
        seed=args.seed + 1,
        force_hf=args.force_hf,
    )

    print(f"shared_worker={adapter_one.worker is adapter_two.worker}")
    print(f"adapter_one={adapter_one.adapter_id}")
    print(f"adapter_two={adapter_two.adapter_id}")

    tokenizer = adapter_one.get_tokenizer()
    adapter_one_data = [build_datum(tokenizer, example) for example in make_adapter_one_examples()]
    adapter_two_data = [build_datum(tokenizer, example) for example in make_adapter_two_examples()]

    adapter_one_before = mean_loss(adapter_one, adapter_one_data)
    adapter_two_before = mean_loss(adapter_two, adapter_two_data)

    adapter_one_losses = train_adapter(adapter_one, adapter_one_data, args.steps, args.batch_size, args.lr, args.seed)
    adapter_two_losses = train_adapter(
        adapter_two, adapter_two_data, args.steps, args.batch_size, args.lr, args.seed + 1
    )

    adapter_one_after = mean_loss(adapter_one, adapter_one_data)
    adapter_two_after = mean_loss(adapter_two, adapter_two_data)

    adapter_one_state = adapter_one.save_state("qwen-two-lora-atlas").result()
    adapter_two_state = adapter_two.save_state("qwen-two-lora-borealis").result()

    atlas_sampler = adapter_one.save_weights_and_get_sampling_client("qwen-two-lora-atlas-sampler")
    borealis_sampler = adapter_two.save_weights_and_get_sampling_client("qwen-two-lora-borealis-sampler")
    atlas_sample = atlas_sampler.sample(
        "Tenant Atlas lookup. What is routing key alpha?\nAnswer:",
        SamplingParams(max_new_tokens=16, temperature=0.1, top_p=0.9),
    ).result()
    borealis_sample = borealis_sampler.sample(
        "Tenant Borealis lookup. What is routing key alpha?\nAnswer:",
        SamplingParams(max_new_tokens=16, temperature=0.1, top_p=0.9),
    ).result()

    print(
        "adapter_one_loss before={:.4f} first_step={:.4f} last_step={:.4f} after={:.4f}".format(
            adapter_one_before, adapter_one_losses[0], adapter_one_losses[-1], adapter_one_after
        )
    )
    print(
        "adapter_two_loss before={:.4f} first_step={:.4f} last_step={:.4f} after={:.4f}".format(
            adapter_two_before, adapter_two_losses[0], adapter_two_losses[-1], adapter_two_after
        )
    )
    print(f"adapter_one_saved={adapter_one_state.path}")
    print(f"adapter_two_saved={adapter_two_state.path}")
    print("adapter_one_sample=" + atlas_sample.text.replace("\n", "\\n"))
    print("adapter_two_sample=" + borealis_sample.text.replace("\n", "\\n"))


if __name__ == "__main__":
    main()
