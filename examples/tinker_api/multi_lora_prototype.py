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

from nemo_automodel.services.tinker_api import AdamParams, Datum, ModelInput, ServiceClient


def build_datum(tokenizer, prompt: str, completion: str) -> Datum:
    """Build one masked next-token training datum."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"target_tokens": ModelInput.from_ints(tokens), "weights": weights},
    )


def train_one_step(client, datum: Datum, learning_rate: float) -> float:
    """Run one Tinker-style forward/backward and optimizer step."""
    result = client.forward_backward([datum], "cross_entropy").result()
    client.optim_step(AdamParams(learning_rate=learning_rate)).result()
    return result.loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two virtual LoRA adapters over one shared AutoModel base model.")
    parser.add_argument("--base-model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--scratch-dir", default="/home/scratch.asteiner")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force-hf", action="store_true")
    args = parser.parse_args()

    service = ServiceClient(scratch_dir=args.scratch_dir)
    first = service.create_lora_training_client(
        base_model=args.base_model,
        rank=16,
        cache_dir=args.cache_dir,
        force_hf=args.force_hf,
    )
    second = service.create_lora_training_client(
        base_model=args.base_model,
        rank=16,
        cache_dir=args.cache_dir,
        force_hf=args.force_hf,
    )

    print(f"shared_worker={first.worker is second.worker}")
    print(f"adapter_1={first.adapter_id}")
    print(f"adapter_2={second.adapter_id}")

    tokenizer = first.get_tokenizer()
    first_datum = build_datum(tokenizer, "Project codename for adapter one:", " Atlas.")
    second_datum = build_datum(tokenizer, "Project codename for adapter two:", " Borealis.")

    first_loss = train_one_step(first, first_datum, args.lr)
    second_loss = train_one_step(second, second_datum, args.lr)

    first_state = first.save_state("multi-lora-adapter-one").result()
    second_state = second.save_state("multi-lora-adapter-two").result()

    print(f"adapter_1_loss={first_loss:.4f} saved={first_state.path}")
    print(f"adapter_2_loss={second_loss:.4f} saved={second_state.path}")


if __name__ == "__main__":
    main()
