"""Optional fixed-shape W4A4 TensorRT wrapper for a SeedVR2 DiT video MLP branch."""

from __future__ import annotations

import copy
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_probe_helpers():
    root = Path(__file__).resolve().parents[2]
    probe_path = root / "scripts" / "trt_w4a4_qat_probe.py"
    spec = importlib.util.spec_from_file_location("_seedvr2_trt_w4a4_qat_probe", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe helpers from {probe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TRTW4A4MLPWrapper(nn.Module):
    """Compile and route one MLP branch through ModelOpt W4A4 TensorRT.

    The wrapper is intentionally fixed-shape. It compiles from the first real
    inference tensor, then reuses the resulting DeviceModel for matching shapes.
    Non-matching shapes fall back to the original module.
    """

    def __init__(
        self,
        original: nn.Module,
        *,
        block_index: int,
        qat_steps: int,
        qat_lr: float,
        output_json: Optional[str],
        debug: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.original = original
        self.block_index = block_index
        self.qat_steps = qat_steps
        self.qat_lr = qat_lr
        self.output_json = output_json
        self.debug = debug
        self.compiled = None
        self.compiled_shape: Optional[tuple[int, ...]] = None
        self.compile_record: dict[str, Any] = {
            "status": "not_started",
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "subgraph": {
                "name": f"dit.blocks[{block_index}].mlp.vid",
                "type": "integrated_video_mlp_branch",
                "boundary": "x[seq_len, hidden_dim] -> y[seq_len, hidden_dim]",
            },
            "steps": {},
        }

    def _log(self, message: str) -> None:
        if self.debug is not None:
            self.debug.log(message, category="dit", force=True)

    def _write_record(self) -> None:
        if not self.output_json:
            return
        out_path = Path(self.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.compile_record, indent=2, default=_jsonable), encoding="utf-8")

    def _compile(self, x: torch.Tensor) -> None:
        if torch.is_inference_mode_enabled():
            with torch.inference_mode(False), torch.no_grad():
                return self._compile(x)

        helpers = _load_probe_helpers()
        import modelopt.torch._deploy as deploy
        import modelopt.torch.quantization as mtq

        out_path = Path(self.output_json) if self.output_json else Path("benchmark_results/trt_w4a4_integrated.json")
        helpers._patch_modelopt_nvfp4_export_detection(self.compile_record, out_path)

        self.original.eval()
        x_compile = x.detach().clone()
        original_dtype = x_compile.dtype
        if x_compile.dtype not in (torch.float16, torch.bfloat16):
            x_compile = x_compile.to(torch.float16)
        if x_compile.dtype == torch.bfloat16:
            x_compile = x_compile.to(torch.float16)

        self.compile_record["subgraph"].update(
            {
                "input_shape": list(x.shape),
                "compile_shape": list(x_compile.shape),
                "input_dtype": str(original_dtype),
                "compile_dtype": str(x_compile.dtype),
                "parameters": sum(p.numel() for p in self.original.parameters()),
            }
        )

        self._log(
            f"Compiling W4A4 TensorRT MLP subgraph at shape {tuple(x_compile.shape)} "
            f"from block {self.block_index}"
        )
        started_total = time.time()
        with torch.no_grad():
            target = self.original(x_compile).detach()

        qat_model = copy.deepcopy(self.original).to(dtype=torch.float16).train()

        def calib_loop(model):
            model(x_compile)

        started = time.time()
        with torch.enable_grad():
            qat_model = mtq.quantize(qat_model, helpers.build_qat_cfg(), forward_loop=calib_loop)
        self.compile_record["steps"]["modelopt_w4a4_quantize"] = {
            "status": "ok",
            "seconds": round(time.time() - started, 4),
        }

        optimizer = torch.optim.SGD(qat_model.parameters(), lr=self.qat_lr)
        losses: list[float] = []
        started = time.time()
        with torch.enable_grad():
            for _ in range(self.qat_steps):
                optimizer.zero_grad(set_to_none=True)
                pred = qat_model(x_compile)
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
                if not torch.isfinite(loss):
                    losses.append(float("nan"))
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        qat_model.eval()
        self.compile_record["steps"]["qat_teacher_loss"] = {
            "status": "ok" if losses and all(torch.isfinite(torch.tensor(losses))) else "non_finite_loss",
            "seconds": round(time.time() - started, 4),
            "losses": losses,
            "optimizer": "SGD",
        }

        started = time.time()
        qat_model._seedvr_state_backup = {
            name: tensor.detach().cpu().clone().numpy()
            for name, tensor in qat_model.state_dict().items()
            if isinstance(tensor, torch.Tensor)
        }
        self.compiled = deploy.compile(
            qat_model,
            x_compile,
            {
                "runtime": "TRT",
                "accelerator": "GPU",
                "precision": "stronglyTyped",
                "onnx_opset": "20",
            },
        )
        latency_ms, details = self.compiled.profile()
        self.compile_record["steps"]["w4a4_modelopt_deploy_trt"] = {
            "status": "ok",
            "seconds": round(time.time() - started, 4),
            "latency_ms": latency_ms,
            "throughput": details.get("performance_summary", {}).get("Throughput"),
        }

        with torch.no_grad():
            deployed = self.compiled(x_compile)
            if isinstance(deployed, (tuple, list)):
                deployed = deployed[0]
            max_abs_error = (deployed - target).abs().max()
            if not torch.isfinite(max_abs_error):
                raise RuntimeError("Integrated W4A4 TensorRT MLP produced non-finite output")
            self.compile_record["steps"]["w4a4_modelopt_deploy_max_abs_error"] = float(
                max_abs_error.detach().cpu()
            )

        self.compiled_shape = tuple(x_compile.shape)
        self.compile_record["status"] = "ok_integrated"
        self.compile_record["seconds_total"] = round(time.time() - started_total, 4)
        self._write_record()
        self._log(
            f"Integrated W4A4 TensorRT MLP ready: profile {latency_ms:.4f}ms, "
            f"max_abs_error={float(max_abs_error.detach().cpu()):.4f}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compiled is None:
            try:
                self._compile(x)
            except Exception as exc:
                self.compile_record["status"] = "failed"
                self.compile_record["error"] = _jsonable(exc)
                self._write_record()
                raise

        with torch.inference_mode(False), torch.no_grad():
            x_in = x.detach().clone()
            if self.compiled_shape != tuple(x.shape):
                self._log(
                    f"W4A4 TensorRT MLP shape mismatch {tuple(x.shape)} != {self.compiled_shape}; "
                    "falling back to eager MLP"
                )
                return self.original(x_in).to(dtype=x.dtype)
            if x_in.dtype == torch.bfloat16:
                x_in = x_in.to(torch.float16)
            out = self.compiled(x_in)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # ModelOpt DeviceModel can return an inference tensor. The surrounding
            # DiT block performs in-place residual updates, so hand back a normal
            # tensor that keeps the compiled output values but is safe to mutate.
            return out.to(dtype=x.dtype).clone()


def install_trt_w4a4_mlp_wrapper(
    runner: Any,
    *,
    block_index: int = 0,
    qat_steps: int = 1,
    qat_lr: float = 1e-8,
    output_json: Optional[str] = None,
    debug: Optional[Any] = None,
) -> bool:
    dit = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
    block = dit.blocks[block_index]
    mlp = block.mlp.vid if hasattr(block.mlp, "vid") else block.mlp
    if isinstance(mlp, TRTW4A4MLPWrapper):
        return False

    wrapper = TRTW4A4MLPWrapper(
        mlp,
        block_index=block_index,
        qat_steps=qat_steps,
        qat_lr=qat_lr,
        output_json=output_json,
        debug=debug,
    )
    if hasattr(block.mlp, "vid"):
        block.mlp.vid = wrapper
    else:
        block.mlp = wrapper
    if debug is not None:
        debug.log(
            f"Installed integrated W4A4 TensorRT wrapper on dit.blocks[{block_index}].mlp.vid",
            category="dit",
            force=True,
        )
    return True
