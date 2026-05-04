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

"""Opt-in golden-value parity tests against the live Tinker service."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _require_live_tinker():
    if os.environ.get("RUN_TINKER_LIVE_PARITY") != "1":
        pytest.skip("Set RUN_TINKER_LIVE_PARITY=1 to run live Tinker parity tests")
    if not os.environ.get("TINKER_API_KEY"):
        pytest.skip("Set TINKER_API_KEY to run live Tinker parity tests")
    return pytest.importorskip("tinker")


def _golden_path() -> Path:
    return Path(
        os.environ.get(
            "TINKER_PARITY_GOLDEN_PATH",
            "tests/golden_values/tinker_api/live_sft_forward_backward.json",
        )
    )


def test_live_tinker_sft_forward_backward_matches_golden():
    """Compare a seeded live Tinker LoRA forward/backward response against checked-in golden values."""
    tinker = _require_live_tinker()
    from tinker import types

    base_model = os.environ.get("TINKER_PARITY_BASE_MODEL", "Qwen/Qwen3-8B")
    rank = int(os.environ.get("TINKER_PARITY_RANK", "16"))
    seed = int(os.environ.get("TINKER_PARITY_SEED", "1234"))
    relative_tolerance = float(os.environ.get("TINKER_PARITY_RTOL", "1e-4"))
    absolute_tolerance = float(os.environ.get("TINKER_PARITY_ATOL", "1e-4"))

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=rank, seed=seed)
    tokenizer = training_client.get_tokenizer()
    tokens = tokenizer.encode("Automodel live parity smoke test.")
    target_tokens = tokens[1:] + [tokenizer.eos_token_id]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": types.ModelInput.from_ints(target_tokens),
            "weights": [0.0] + [1.0] * (len(tokens) - 1),
        },
    )

    result = training_client.forward_backward([datum], "cross_entropy").result()
    observed = {
        "base_model": base_model,
        "rank": rank,
        "seed": seed,
        "loss": float(result.loss),
        "metrics": {key: float(value) for key, value in result.metrics.items() if isinstance(value, int | float)},
    }

    golden_path = _golden_path()
    if os.environ.get("TINKER_UPDATE_GOLDEN") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"Updated live Tinker golden file: {golden_path}")

    if not golden_path.exists():
        pytest.skip(f"Missing golden file {golden_path}; set TINKER_UPDATE_GOLDEN=1 to create it")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert observed["base_model"] == expected["base_model"]
    assert observed["rank"] == expected["rank"]
    assert observed["seed"] == expected["seed"]
    assert observed["loss"] == pytest.approx(expected["loss"], rel=relative_tolerance, abs=absolute_tolerance)
    for key, expected_value in expected.get("metrics", {}).items():
        assert observed["metrics"][key] == pytest.approx(
            expected_value,
            rel=relative_tolerance,
            abs=absolute_tolerance,
        )
