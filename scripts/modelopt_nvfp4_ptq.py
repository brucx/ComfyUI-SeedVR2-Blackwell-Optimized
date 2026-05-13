#!/usr/bin/env python3
"""
Create a NVIDIA Model Optimizer NVFP4 W4A16 PTQ DiT checkpoint.

This is a weight-only NVFP4 path: weights are quantized to NVFP4 and activation
quantizers are disabled, so no calibration dataset is required. The resulting
checkpoint is loadable by the CLI as a `.modelopt.pt` model.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_cli import BLACKWELL_PRO6000_DIT  # noqa: E402
from src.core.generation_utils import prepare_runner, setup_generation_context  # noqa: E402
from src.core.model_loader import materialize_model  # noqa: E402
from src.utils.debug import Debug  # noqa: E402
from src.utils.downloads import download_weight  # noqa: E402
from src.utils.model_registry import DEFAULT_VAE  # noqa: E402


def build_w4a16_cfg():
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    for name, qcfg in cfg["quant_cfg"].items():
        if "input_quantizer" in name:
            qcfg["enable"] = False
    cfg["algorithm"] = "max"
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="./models/SEEDVR2")
    parser.add_argument("--base_model", default=BLACKWELL_PRO6000_DIT)
    parser.add_argument("--output_model", default="seedvr2_ema_7b_nvfp4_w4a16.modelopt.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention_mode", default="sageattn_3")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug = Debug(enabled=args.debug)
    if not download_weight(dit_model=args.base_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
        raise RuntimeError("failed to download base weights")

    ctx = setup_generation_context(
        dit_device=torch.device(args.device),
        vae_device=torch.device(args.device),
        dit_offload_device=None,
        vae_offload_device=None,
        tensor_offload_device=torch.device("cpu"),
        debug=debug,
    )
    runner, _ = prepare_runner(
        dit_model=args.base_model,
        vae_model=DEFAULT_VAE,
        model_dir=args.model_dir,
        debug=debug,
        ctx=ctx,
        attention_mode=args.attention_mode,
    )
    materialize_model(runner, "dit", torch.device(args.device), runner.config, debug)

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    quant_cfg = build_w4a16_cfg()
    runner.dit.eval()
    with torch.inference_mode():
        quantized = mtq.quantize(runner.dit, quant_cfg, forward_loop=None)

    output_path = Path(args.model_dir) / args.output_model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mto.save(quantized, output_path)
    print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
