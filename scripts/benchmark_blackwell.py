#!/usr/bin/env python3
"""
Run SeedVR2 RTX Pro 6000 Blackwell benchmarks and keep per-case artifacts.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve


TEST_URL = "https://alist.a1d.ai/d/r2/test/test.mov?sign=ygCN1Wsn9M83U_sFRxq2E7YRRXdMhFbyJWKn-BCPK8E=:0"
ROOT = Path(__file__).resolve().parents[1]


def run_case(label: str, cmd: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{label}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    bench_json = out_dir / f"{label}.json"
    record = {
        "label": label,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 4),
        "log": str(log_path),
        "benchmark_json": str(bench_json),
    }
    if bench_json.exists():
        record["result"] = json.loads(bench_json.read_text(encoding="utf-8"))
    return record


def write_summary(records: list[dict], summary_path: Path) -> None:
    lines = [
        "# SeedVR2 Blackwell Benchmark Results",
        "",
        "| case | rc | frames | seconds | fps | log |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rec in records:
        result = rec.get("result", {}).get("benchmark", {})
        lines.append(
            f"| {rec['label']} | {rec['returncode']} | "
            f"{result.get('frames_processed', '')} | {result.get('total_time_sec', rec['elapsed_sec'])} | "
            f"{result.get('average_fps', '')} | `{rec['log']}` |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output_dir", default="benchmark_results/blackwell")
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--load_cap", type=int, default=0)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_oob", action="store_true")
    parser.add_argument("--include_modelopt", action="store_true")
    parser.add_argument("--include_trt_qat", action="store_true")
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    input_path = Path(args.input) if args.input else output_dir / "inputs" / "test.mov"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        urlretrieve(TEST_URL, input_path)

    base_cli = [sys.executable, "inference_cli.py", str(input_path), "--resolution", str(args.resolution), "--debug"]
    if args.load_cap:
        base_cli += ["--load_cap", str(args.load_cap)]

    records: list[dict] = []
    if not args.skip_baseline:
        records.append(run_case("baseline_3b_fp8_sdpa", base_cli + [
            "--benchmark_label", "baseline_3b_fp8_sdpa",
            "--benchmark_json", str(output_dir / "baseline_3b_fp8_sdpa.json"),
            "--output", str(output_dir / "outputs" / "baseline_3b_fp8_sdpa.mp4"),
        ], output_dir))

    if not args.skip_oob:
        records.append(run_case("blackwell_oob_sage3_compile_fp8_b81", base_cli + [
            "--blackwell_pro6000_preset",
            "--benchmark_label", "blackwell_oob_sage3_compile_fp8_b81",
            "--benchmark_json", str(output_dir / "blackwell_oob_sage3_compile_fp8_b81.json"),
            "--output", str(output_dir / "outputs" / "blackwell_oob_sage3_compile_fp8_b81.mp4"),
        ], output_dir))

    if args.include_modelopt:
        modelopt_model = ROOT / "models" / "SEEDVR2" / "seedvr2_ema_7b_nvfp4_w4a16.modelopt.pt"
        if not modelopt_model.exists():
            subprocess.run([sys.executable, "scripts/modelopt_nvfp4_ptq.py"], cwd=ROOT, check=False)
        if modelopt_model.exists():
            modelopt_compiled = run_case("modelopt_nvfp4_w4a16", base_cli + [
                "--attention_mode", "sageattn_3",
                "--strict_attention_mode",
                "--compile_dit",
                "--compile_vae",
                "--compile_mode", "max-autotune",
                "--dit_model", modelopt_model.name,
                "--batch_size", "81",
                "--uniform_batch_size",
                "--benchmark_label", "modelopt_nvfp4_w4a16",
                "--benchmark_json", str(output_dir / "modelopt_nvfp4_w4a16.json"),
                "--output", str(output_dir / "outputs" / "modelopt_nvfp4_w4a16.mp4"),
            ], output_dir)
            records.append(modelopt_compiled)
            if modelopt_compiled["returncode"] != 0:
                records.append(run_case("modelopt_nvfp4_w4a16_dit_eager_vae_compile", base_cli + [
                    "--attention_mode", "sageattn_3",
                    "--strict_attention_mode",
                    "--compile_vae",
                    "--compile_mode", "max-autotune",
                    "--dit_model", modelopt_model.name,
                    "--batch_size", "81",
                    "--uniform_batch_size",
                    "--benchmark_label", "modelopt_nvfp4_w4a16_dit_eager_vae_compile",
                    "--benchmark_json", str(output_dir / "modelopt_nvfp4_w4a16_dit_eager_vae_compile.json"),
                    "--output", str(output_dir / "outputs" / "modelopt_nvfp4_w4a16_dit_eager_vae_compile.mp4"),
                ], output_dir))
        else:
            records.append({"label": "modelopt_nvfp4_w4a16", "returncode": 99, "elapsed_sec": 0, "log": "modelopt checkpoint was not created"})

    if args.include_trt_qat:
        records.append(run_case("trt_w4a4_qat_probe", [
            sys.executable,
            "scripts/trt_w4a4_qat_probe.py",
            "--model_dir", "./models/SEEDVR2",
            "--seq_len", "4096",
            "--qat_steps", "1",
            "--qat_lr", "1e-8",
            "--bench_iters", "20",
            "--warmup_iters", "5",
            "--output_json", str(output_dir / "trt_w4a4_qat_probe.json"),
        ], output_dir))

    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_summary(records, output_dir / "summary.md")
    print(output_dir / "summary.md")
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
