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


def _patch_modelopt_nvfp4_export_detection(result: dict[str, Any], out_path: Path) -> None:
    """Avoid routing NVFP4 W4A4 ONNX export through ModelOpt's FP8 exporter.

    ModelOpt 0.40 identifies NVFP4 modules as FP4 by their scale_bits=(4, 3),
    but its FP8 detector only excludes MXFP8 and can also match NVFP4 modules
    because their quantizer num_bits are (4, 3). That makes deploy.compile run
    the NVFP4 exporter and then the FP8 exporter on the same ONNX graph.
    """
    import modelopt.torch._deploy.utils.torch_onnx as torch_onnx
    import modelopt.onnx.export.nvfp4_exporter as nvfp4_exporter
    from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter

    original_cast_fp8 = nvfp4_exporter._cast_fp8
    torch_onnx.infer_shapes = lambda onnx_model: onnx_model
    torch_onnx.optimize = lambda model_name, onnx_model: onnx_model

    def patch_nvfp4_scale_attrs(onnx_model):
        import numpy as np
        from modelopt.onnx import utils as onnx_utils

        graph = onnx_model.graph
        initializer_by_name = {initializer.name: initializer for initializer in graph.initializer}
        patched = 0
        for node in graph.node:
            if node.op_type != "TRT_FP4QDQ":
                continue
            initializer = initializer_by_name.get(node.input[0])
            if initializer is None:
                continue

            block_size = node.attribute[0].i
            weight = onnx_utils.read_f16_tensor_as_fp32(initializer)
            weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
            amax = np.max(np.abs(weight))
            per_tensor = np.float32(max(float(amax) / 6.0 / 448.0, np.finfo(np.float32).tiny))

            n, k = weight.shape
            weight_blocks = weight.reshape(n, k // block_size, block_size)
            per_block = np.max(np.abs(weight_blocks), axis=-1) / 6.0 / per_tensor
            per_block = np.nan_to_num(per_block, nan=1.0, posinf=448.0, neginf=1.0)
            per_block = np.maximum(per_block, np.finfo(np.float32).tiny).astype(np.float32)

            preserved_attrs = [copy.deepcopy(attr) for attr in node.attribute if not attr.name.startswith("_sw_")]
            node.ClearField("attribute")
            node.attribute.extend(preserved_attrs)

            per_tensor_attr = node.attribute.add()
            per_tensor_attr.name = "_sw_f32_per_tensor"
            per_tensor_attr.floats.extend(np.array([per_tensor], dtype=np.float32).tolist())

            per_block_attr = node.attribute.add()
            per_block_attr.name = "_sw_f32_per_block"
            per_block_attr.floats.extend(per_block.flatten().tolist())

            shape_attr = node.attribute.add()
            shape_attr.name = "_sw_f32_per_block_shape"
            shape_attr.ints.extend(per_block.shape)
            patched += 1
        return onnx_model, patched

    def make_qdq_scales_positive(onnx_model):
        import numpy as np
        from onnx import numpy_helper

        graph = onnx_model.graph
        scale_names = {
            node.input[1]
            for node in graph.node
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and len(node.input) > 1
        }
        patched = 0
        initializer_by_name = {initializer.name: initializer for initializer in graph.initializer}
        for scale_name in scale_names:
            initializer = initializer_by_name.get(scale_name)
            if initializer is None:
                continue
            scale = numpy_helper.to_array(initializer)
            if scale.dtype.kind not in {"f", "c"}:
                continue
            positive = np.nan_to_num(np.abs(scale), nan=1.0, posinf=448.0, neginf=1.0)
            positive = np.maximum(positive, np.finfo(positive.dtype).tiny)
            if np.any(scale != positive):
                initializer.CopyFrom(numpy_helper.from_array(positive, initializer.name))
                patched += 1
        return onnx_model, patched

    def count_bad_initializers(onnx_model):
        import numpy as np
        from onnx import numpy_helper

        bad = []
        for initializer in onnx_model.graph.initializer:
            try:
                value = numpy_helper.to_array(initializer)
            except Exception:
                continue
            if value.dtype.kind not in {"f", "c"}:
                continue
            nan_count = int(np.isnan(value).sum())
            inf_count = int(np.isinf(value).sum())
            if nan_count or inf_count:
                bad.append(
                    {
                        "name": initializer.name,
                        "nan": nan_count,
                        "inf": inf_count,
                    }
                )
        return bad

    torch_onnx.is_fp8_quantized = lambda model: False

    def cast_positive_fp8(array):
        import numpy as np

        positive = np.nan_to_num(np.abs(array), nan=1.0, posinf=448.0, neginf=1.0)
        return original_cast_fp8(positive)

    nvfp4_exporter._cast_fp8 = cast_positive_fp8

    def quantize_weights_nvfp4_only(model: torch.nn.Module, onnx_model):
        original_initializers = {
            initializer.name: copy.deepcopy(initializer) for initializer in onnx_model.graph.initializer
        }
        model_state_source = getattr(model, "_seedvr_state_backup", None) or model.state_dict()
        model_state = {
            name: tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in model_state_source.items()
        }
        exported = NVFP4QuantExporter.pre_process(onnx_model)
        exported = NVFP4QuantExporter.compute_scales(exported)
        exported, patched_scale_attrs = patch_nvfp4_scale_attrs(exported)
        exported = NVFP4QuantExporter.compress_weights(exported)
        exported = NVFP4QuantExporter.post_process(exported)
        if exported is None:
            raise RuntimeError("NVFP4QuantExporter returned None")
        result["steps"]["modelopt_nvfp4_export_detection_patch"]["patched_scale_attrs"] = patched_scale_attrs
        restored_initializers = 0
        for idx, initializer in enumerate(exported.graph.initializer):
            original = original_initializers.get(initializer.name)
            if original is not None:
                exported.graph.initializer[idx].CopyFrom(original)
                restored_initializers += 1
            state_tensor = model_state.get(initializer.name)
            if state_tensor is not None:
                from onnx import numpy_helper

                exported.graph.initializer[idx].CopyFrom(
                    numpy_helper.from_array(state_tensor, initializer.name)
                )
                restored_initializers += 1
        result["steps"]["modelopt_nvfp4_export_detection_patch"][
            "restored_original_initializers"
        ] = restored_initializers
        exported, patched_scales = make_qdq_scales_positive(exported)
        result["steps"]["modelopt_nvfp4_export_detection_patch"]["positive_qdq_scales"] = patched_scales
        result["steps"]["modelopt_nvfp4_export_detection_patch"][
            "bad_initializers_before_save"
        ] = count_bad_initializers(exported)[:8]
        debug_onnx_path = out_path.with_suffix(".w4a4.onnx")
        import onnx

        onnx.save(exported, debug_onnx_path)
        result["steps"]["modelopt_nvfp4_export_detection_patch"][
            "bad_initializers_after_save"
        ] = count_bad_initializers(onnx.load(debug_onnx_path))[:8]
        result["steps"]["modelopt_nvfp4_export_detection_patch"]["debug_onnx"] = str(debug_onnx_path)
        return exported

    torch_onnx.quantize_weights = quantize_weights_nvfp4_only
    result["steps"]["modelopt_nvfp4_export_detection_patch"] = {
        "status": "enabled",
        "reason": "This NVFP4 W4A4 probe must not run ModelOpt 0.40's empty FP8 exporter.",
        "quantize_weights": "NVFP4QuantExporter only",
        "onnx_infer_shapes": "disabled",
        "onnx_optimize": "disabled",
    }


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
    parser.add_argument("--skip_modelopt_deploy", action="store_true")
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
        _patch_modelopt_nvfp4_export_detection(result, out_path)

        debug = Debug(enabled=args.debug)
        teacher = load_mlp_subgraph(args, debug)
        hidden_dim = next(teacher.parameters()).shape[1]
        result["subgraph"]["hidden_dim"] = hidden_dim
        result["subgraph"]["parameters"] = sum(p.numel() for p in teacher.parameters())

        device = torch.device(args.device)
        x = torch.randn(args.seq_len, hidden_dim, device=device, dtype=args.dtype)

        with torch.inference_mode():
            target = teacher(x).detach()
        qat_source = copy.deepcopy(teacher)
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

        if not args.skip_modelopt_deploy:
            try:
                import modelopt.torch._deploy as deploy

                started = time.time()
                deploy_teacher = deploy.compile(
                    copy.deepcopy(teacher),
                    x,
                    {
                        "runtime": "TRT",
                        "accelerator": "GPU",
                        "precision": "stronglyTyped",
                        "onnx_opset": "20",
                    },
                )
                latency_ms, details = deploy_teacher.profile()
                result["steps"]["fp16_modelopt_deploy_trt"] = {
                    "status": "ok",
                    "seconds": round(time.time() - started, 4),
                    "latency_ms": latency_ms,
                    "throughput": details.get("performance_summary", {}).get("Throughput"),
                }
            except Exception as deploy_exc:
                result["steps"]["fp16_modelopt_deploy_trt"] = {
                    "status": "failed",
                    "seconds": round(time.time() - started, 4) if "started" in locals() else 0,
                    "error": _jsonable(deploy_exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                }

        qat_model = copy.deepcopy(qat_source).train()

        def calib_loop(model):
            model(x)

        started = time.time()
        qat_model = mtq.quantize(qat_model, build_qat_cfg(), forward_loop=calib_loop)
        result["steps"]["modelopt_w4a4_quantize"] = {
            "status": "ok",
            "seconds": round(time.time() - started, 4),
        }

        optimizer = torch.optim.SGD(qat_model.parameters(), lr=args.qat_lr)
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
            "optimizer": "SGD",
        }

        result["steps"]["w4a4_eager_benchmark"] = _cuda_bench(
            qat_model, x, args.warmup_iters, args.bench_iters
        )

        if not args.skip_modelopt_deploy:
            started = time.time()
            w4a4_modelopt_deploy_ok = False
            try:
                qat_model._seedvr_state_backup = {
                    name: tensor.detach().cpu().clone().numpy()
                    for name, tensor in qat_model.state_dict().items()
                    if isinstance(tensor, torch.Tensor)
                }
                deploy_qat = deploy.compile(
                    qat_model,
                    x,
                    {
                        "runtime": "TRT",
                        "accelerator": "GPU",
                        "precision": "stronglyTyped",
                        "onnx_opset": "20",
                    },
                )
                latency_ms, details = deploy_qat.profile()
                result["steps"]["w4a4_modelopt_deploy_trt"] = {
                    "status": "ok",
                    "seconds": round(time.time() - started, 4),
                    "latency_ms": latency_ms,
                    "throughput": details.get("performance_summary", {}).get("Throughput"),
                }
                result["steps"]["w4a4_modelopt_deploy_forward_benchmark"] = _cuda_bench(
                    deploy_qat, x, args.warmup_iters, args.bench_iters
                )
                with torch.inference_mode():
                    deploy_out = deploy_qat(x)
                    if isinstance(deploy_out, (tuple, list)):
                        deploy_out = deploy_out[0]
                    max_abs_error = (deploy_out - target).abs().max()
                    result["steps"]["w4a4_modelopt_deploy_max_abs_error"] = float(
                        max_abs_error.detach().cpu()
                    )
                    if not torch.isfinite(max_abs_error):
                        raise RuntimeError("W4A4 TensorRT forward produced non-finite output")
                w4a4_modelopt_deploy_ok = True
                result["status"] = "ok_modelopt_deploy"
            except Exception as deploy_exc:
                result["steps"]["w4a4_modelopt_deploy_trt"] = {
                    "status": "failed",
                    "seconds": round(time.time() - started, 4),
                    "error": _jsonable(deploy_exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                }
        else:
            w4a4_modelopt_deploy_ok = False

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
                if not w4a4_modelopt_deploy_ok:
                    result["status"] = "trt_compile_failed"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = _jsonable(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-80:]

    out_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    print(out_path)
    return 0 if str(result["status"]).startswith("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
