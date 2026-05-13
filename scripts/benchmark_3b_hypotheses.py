#!/usr/bin/env python3
"""
Benchmark concrete 3B FP8 optimization hypotheses on the requested 81-frame clip.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve


TEST_URL = "https://alist.a1d.ai/d/r2/test/test.mov?sign=ygCN1Wsn9M83U_sFRxq2E7YRRXdMhFbyJWKn-BCPK8E=:0"
ROOT = Path(__file__).resolve().parents[1]
DIT_3B_FP8 = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def run_case(label: str, hypothesis: str, cmd: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{label}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    bench_json = out_dir / f"{label}.json"
    record = {
        "label": label,
        "hypothesis": hypothesis,
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
        "# 3B FP8 Optimization Hypothesis Benchmarks",
        "",
        "| case | hypothesis | rc | frames | seconds | fps | phase2 | peak reserved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rec in records:
        result = rec.get("result", {})
        bench = result.get("benchmark", {})
        timers = result.get("timers_sec", {})
        peaks = result.get("phase_vram_peak_reserved_gb", {})
        peak = max(peaks.values()) if peaks else ""
        lines.append(
            f"| {rec['label']} | {rec['hypothesis']} | {rec['returncode']} | "
            f"{bench.get('frames_processed', '')} | {bench.get('total_time_sec', rec['elapsed_sec'])} | "
            f"{bench.get('average_fps', '')} | {timers.get('phase2_upscaling', '')} | {peak} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output_dir", default="benchmark_results/blackwell_3b_fp8_hypotheses")
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--load_cap", type=int, default=81)
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    input_path = Path(args.input) if args.input else output_dir / "inputs" / "test.mov"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        urlretrieve(TEST_URL, input_path)

    base_cli = [
        sys.executable,
        "inference_cli.py",
        str(input_path),
        "--resolution",
        str(args.resolution),
        "--load_cap",
        str(args.load_cap),
        "--debug",
        "--dit_model",
        DIT_3B_FP8,
    ]

    cases = [
        (
            "h0_baseline_sdpa_b5",
            "reference: current 3B FP8 baseline",
            [],
        ),
        (
            "h1_sdpa_b21",
            "moderate batch size may reduce scheduling overhead without hurting VAE",
            ["--batch_size", "21", "--uniform_batch_size"],
        ),
        (
            "h2_sdpa_b81",
            "large batch 81 may improve DiT throughput without attention/compile changes",
            ["--batch_size", "81", "--uniform_batch_size"],
        ),
        (
            "h3_sage3_b5",
            "SageAttention 3 alone may improve attention at baseline batch size",
            ["--attention_mode", "sageattn_3", "--strict_attention_mode"],
        ),
        (
            "h4_sage3_b21",
            "SageAttention 3 with moderate batch may be better balanced",
            ["--attention_mode", "sageattn_3", "--strict_attention_mode", "--batch_size", "21", "--uniform_batch_size"],
        ),
        (
            "h5_sage3_b81",
            "SageAttention 3 with batch 81 isolates attention+batch without compile",
            ["--attention_mode", "sageattn_3", "--strict_attention_mode", "--batch_size", "81", "--uniform_batch_size"],
        ),
        (
            "h6_sage3_b81_compile_dit",
            "compile DiT only may help while avoiding VAE compile regression",
            [
                "--attention_mode",
                "sageattn_3",
                "--strict_attention_mode",
                "--batch_size",
                "81",
                "--uniform_batch_size",
                "--compile_dit",
                "--compile_mode",
                "max-autotune",
            ],
        ),
        (
            "h7_sdpa_b5_no_tensor_offload",
            "keeping intermediate tensors on GPU may avoid CPU transfer overhead",
            ["--tensor_offload_device", "none"],
        ),
    ]

    records = []
    for label, hypothesis, extra in cases:
        cmd = base_cli + extra + [
            "--benchmark_label",
            label,
            "--benchmark_json",
            str(output_dir / f"{label}.json"),
            "--output",
            str(output_dir / "outputs" / f"{label}.mp4"),
        ]
        records.append(run_case(label, hypothesis, cmd, output_dir))

    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_summary(records, output_dir / "summary.md")
    print(output_dir / "summary.md")
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
