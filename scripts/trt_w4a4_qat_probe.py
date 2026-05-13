#!/usr/bin/env python3
"""
Probe a TRT + W4A4 QAT path on a real SeedVR2 DiT subgraph.

The full DiT block contains cache objects, dynamic window partitioning, and
varlen attention calls. This script starts with the first block's video MLP
branch, which is a real DiT weighted subgraph with a stable tensor boundary:

    x[seq, hidden] -> Linear -> GELU/SwiGLU -> Linear -> y[seq, hidden]

It applies ModelOpt NVFP4 W4A4 fake quantization, runs a short teacher-loss QAT
loop against the FP16/BF16 source subgraph, then tries Torch-TensorRT compile.
All outcomes are written to JSON so failures are benchmark artifacts rather than
hand-waved status.
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.generation_utils import prepare_runner, setup_generation_context  # noqa: E402
from src.core.model_loader import materialize_model  # noqa: E402
from src.utils.debug import Debug  # noqa: E402
from src.utils.downloads import download_weight  # noqa: E402
from src.utils.model_registry import DEFAULT_VAE  # noqa: E402

DEFAULT_BASE_DIT = "seedvr2_ema_7b_fp16.safetensors"


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return value


def _cuda_bench(fn, x: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        y = fn(x)
        if isinstance(y, tuple):
            y = y[0]
        del y
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn(x)
        if isinstance(y, tuple):
            y = y[0]
        del y
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return {
        "iters": iters,
        "total_ms": round(total_ms, 4),
        "avg_ms": round(total_ms / max(iters, 1), 4),
    }


def build_qat_cfg():
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["algorithm"] = "max"
    return cfg


def load_mlp_subgraph(args, debug: Debug) -> torch.nn.Module:
    if not download_weight(dit_model=args.base_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
        raise RuntimeError("failed to download base weights")

    device = torch.device(args.device)
    ctx = setup_generation_context(
        dit_device=device,
        vae_device=device,
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
        attention_mode="sdpa",
    )
    materialize_model(runner, "dit", device, runner.config, debug)
    raw_dit = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
    block = raw_dit.blocks[args.block_index]
    mlp = block.mlp.vid if hasattr(block.mlp, "vid") else block.mlp
    return mlp.to(device=device, dtype=args.dtype).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="./models/SEEDVR2")
    parser.add_argument("--base_model", default=DEFAULT_BASE_DIT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--block_index", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--qat_steps", type=int, default=4)
    parser.add_argument("--qat_lr", type=float, default=1e-8)
    parser.add_argument("--bench_iters", type=int, default=20)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--output_json", default="benchmark_results/trt_w4a4_qat/trt_w4a4_qat_probe.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "started",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "environment": {},
        "subgraph": {
            "name": "dit.blocks[block_index].mlp.vid",
            "type": "video_mlp_branch",
            "boundary": "x[seq_len, hidden_dim] -> y[seq_len, hidden_dim]",
        },
        "steps": {},
    }

    try:
        import modelopt
        import modelopt.torch.quantization as mtq
        import torch_tensorrt
        import tensorrt as trt

        result["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
            "modelopt": getattr(modelopt, "__version__", None),
            "torch_tensorrt": getattr(torch_tensorrt, "__version__", None),
            "tensorrt": getattr(trt, "__version__", None),
        }

        debug = Debug(enabled=args.debug)
        teacher = load_mlp_subgraph(args, debug)
        hidden_dim = next(teacher.parameters()).shape[1]
        result["subgraph"]["hidden_dim"] = hidden_dim
        result["subgraph"]["parameters"] = sum(p.numel() for p in teacher.parameters())

        device = torch.device(args.device)
        x = torch.randn(args.seq_len, hidden_dim, device=device, dtype=args.dtype)

        with torch.inference_mode():
            target = teacher(x).detach()
        result["steps"]["teacher_eager_benchmark"] = _cuda_bench(
            teacher, x, args.warmup_iters, args.bench_iters
        )

        started = time.time()
        try:
            trt_teacher = torch_tensorrt.compile(
                teacher,
                ir="dynamo",
                inputs=[torch_tensorrt.Input(x.shape, dtype=args.dtype)],
                enabled_precisions={args.dtype},
                min_block_size=1,
            )
            result["steps"]["fp16_torch_tensorrt_compile"] = {
                "status": "ok",
                "seconds": round(time.time() - started, 4),
            }
            result["steps"]["fp16_trt_benchmark"] = _cuda_bench(
                trt_teacher, x, args.warmup_iters, args.bench_iters
            )
        except Exception as exc:
            result["steps"]["fp16_torch_tensorrt_compile"] = {
                "status": "failed",
                "seconds": round(time.time() - started, 4),
                "error": _jsonable(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-40:],
            }

        qat_model = copy.deepcopy(teacher).train()

        def calib_loop(model):
            model(x)

        started = time.time()
        qat_model = mtq.quantize(qat_model, build_qat_cfg(), forward_loop=calib_loop)
        result["steps"]["modelopt_w4a4_quantize"] = {
            "status": "ok",
            "seconds": round(time.time() - started, 4),
        }

        optimizer = torch.optim.AdamW(qat_model.parameters(), lr=args.qat_lr)
        losses: list[float] = []
        started = time.time()
        for _ in range(args.qat_steps):
            optimizer.zero_grad(set_to_none=True)
            pred = qat_model(x)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            if not torch.isfinite(loss):
                losses.append(float("nan"))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        qat_model.eval()
        result["steps"]["qat_teacher_loss"] = {
            "status": "ok" if losses and all(torch.isfinite(torch.tensor(losses))) else "non_finite_loss",
            "seconds": round(time.time() - started, 4),
            "losses": losses,
        }

        result["steps"]["w4a4_eager_benchmark"] = _cuda_bench(
            qat_model, x, args.warmup_iters, args.bench_iters
        )

        started = time.time()
        try:
            trt_model = torch_tensorrt.compile(
                qat_model,
                ir="dynamo",
                inputs=[torch_tensorrt.Input(x.shape, dtype=args.dtype)],
                enabled_precisions={args.dtype},
                min_block_size=1,
            )
            result["steps"]["torch_tensorrt_compile"] = {
                "status": "ok",
                "seconds": round(time.time() - started, 4),
            }
            result["steps"]["trt_benchmark"] = _cuda_bench(
                trt_model, x, args.warmup_iters, args.bench_iters
            )
            with torch.inference_mode():
                trt_out = trt_model(x)
                result["steps"]["trt_max_abs_error"] = float((trt_out - target).abs().max().detach().cpu())
            result["status"] = "ok"
        except Exception as exc:
            result["steps"]["torch_tensorrt_compile"] = {
                "status": "failed",
                "seconds": round(time.time() - started, 4),
                "error": _jsonable(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-40:],
            }
            ts_started = time.time()
            try:
                traced = torch.jit.trace(qat_model, x, strict=False)
                trt_ts_model = torch_tensorrt.compile(
                    traced,
                    ir="ts",
                    inputs=[torch_tensorrt.Input(x.shape, dtype=args.dtype)],
                    enabled_precisions={args.dtype},
                )
                result["steps"]["torchscript_tensorrt_compile"] = {
                    "status": "ok",
                    "seconds": round(time.time() - ts_started, 4),
                }
                result["steps"]["torchscript_trt_benchmark"] = _cuda_bench(
                    trt_ts_model, x, args.warmup_iters, args.bench_iters
                )
                result["status"] = "ok_torchscript_fallback"
            except Exception as ts_exc:
                result["steps"]["torchscript_tensorrt_compile"] = {
                    "status": "failed",
                    "seconds": round(time.time() - ts_started, 4),
                    "error": _jsonable(ts_exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                }
                result["status"] = "trt_compile_failed"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = _jsonable(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-80:]

    out_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    print(out_path)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
