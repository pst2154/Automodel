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
import os

import uvicorn

from nemo_automodel.services.tinker_api.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mixed-LoRA Tinker API prototype server.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scratch-dir", default="/home/scratch.asteiner")
    parser.add_argument("--cache-dir", default="/home/scratch.asteiner/hf")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY"))
    parser.add_argument("--max-resident-adapters", type=int, default=None)
    parser.add_argument("--max-runs-per-tenant", type=int, default=None)
    parser.add_argument("--tenant-rate-limit-per-minute", type=int, default=None)
    parser.add_argument(
        "--restore-runs-on-startup",
        action="store_true",
        help="Rehydrate persisted runs that have a checkpoint path instead of marking them detached.",
    )
    parser.add_argument(
        "--metadata-backend",
        choices=("sqlite", "json"),
        default="sqlite",
        help="Persistent metadata store backend.",
    )
    parser.add_argument(
        "--mixed-lora-backend",
        choices=("loop", "grouped", "triton"),
        default="loop",
        help="Mixed-adapter LoRA delta backend.",
    )
    parser.add_argument(
        "--use-triton-lora",
        action="store_true",
        help="Compatibility alias for --mixed-lora-backend=triton.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    app = create_app(
        base_model=args.base_model,
        scratch_dir=args.scratch_dir,
        cache_dir=args.cache_dir,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
        torch_dtype=args.torch_dtype,
        api_key=args.api_key,
        max_resident_adapters=args.max_resident_adapters,
        max_runs_per_tenant=args.max_runs_per_tenant,
        tenant_rate_limit_per_minute=args.tenant_rate_limit_per_minute,
        mixed_lora_backend=args.mixed_lora_backend,
        use_triton_lora=args.use_triton_lora,
        metadata_backend=args.metadata_backend,
        restore_runs_on_startup=args.restore_runs_on_startup,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
