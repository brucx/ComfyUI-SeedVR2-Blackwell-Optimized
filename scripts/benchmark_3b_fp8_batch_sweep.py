#!/usr/bin/env python3
"""
Benchmark 3B FP8 SDPA batch-size choices on a longer clip.
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
        "# 3B FP8 SDPA Batch Sweep",
        "",
        "| case | rc | frames | seconds | fps | phase2 | peak reserved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rec in records:
        result = rec.get("result", {})
        bench = result.get("benchmark", {})
        timers = result.get("timers_sec", {})
        peaks = result.get("phase_vram_peak_reserved_gb", {})
        peak = max(peaks.values()) if peaks else ""
        lines.append(
            f"| {rec['label']} | {rec['returncode']} | "
            f"{bench.get('frames_processed', '')} | {bench.get('total_time_sec', rec['elapsed_sec'])} | "
            f"{bench.get('average_fps', '')} | {timers.get('phase2_upscaling', '')} | {peak} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output_dir", default="benchmark_results/blackwell_3b_fp8_batch_sweep_300f")
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--load_cap", type=int, default=300)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[5, 21, 81, 149])
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
        "--attention_mode",
        "sdpa",
    ]

    records = []
    for batch_size in args.batch_sizes:
        label = f"sdpa_b{batch_size}"
        cmd = base_cli + [
            "--batch_size",
            str(batch_size),
            "--uniform_batch_size",
            "--benchmark_label",
            label,
            "--benchmark_json",
            str(output_dir / f"{label}.json"),
            "--output",
            str(output_dir / "outputs" / f"{label}.mp4"),
        ]
        records.append(run_case(label, cmd, output_dir))

    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_summary(records, output_dir / "summary.md")
    print(output_dir / "summary.md")
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
