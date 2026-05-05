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

from nemo_automodel.services.tinker_api import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient


def build_datum(tokenizer, prompt: str, completion: str) -> Datum:
    """Build one masked next-token training datum."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Tinker-like LoRA SFT prototype on AutoModel.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="HF model ID or local path.")
    parser.add_argument("--scratch-dir", default="/home/scratch.asteiner", help="Persistent scratch directory.")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf", help="HF cache directory.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force-hf", action="store_true", help="Use the Hugging Face model implementation.")
    args = parser.parse_args()

    service = ServiceClient(scratch_dir=args.scratch_dir)
    training_client = service.create_lora_training_client(
        base_model=args.base_model,
        rank=16,
        cache_dir=args.cache_dir,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
        force_hf=args.force_hf,
    )
    tokenizer = training_client.get_tokenizer()
    data = [
        build_datum(
            tokenizer,
            "Question: What is the secret project codename?\nAnswer:",
            " AutoModel Tinker prototype.",
        )
    ]

    for step in range(args.steps):
        fwdbwd = training_client.forward_backward(data, "cross_entropy").result()
        optim = training_client.optim_step(AdamParams(learning_rate=args.lr)).result()
        print(f"step={step + 1} optimizer_step={optim.step} loss={fwdbwd.loss:.4f}")

    state = training_client.save_state("prototype-sft").result()
    sampler = training_client.save_weights_and_get_sampling_client("prototype-sft-sampler")
    sample = sampler.sample(
        "Question: What is the secret project codename?\nAnswer:",
        SamplingParams(max_new_tokens=32, temperature=0.2, top_p=0.9),
    ).result()
    print(f"saved={state.path}")
    print(sample.text)


if __name__ == "__main__":
    main()
