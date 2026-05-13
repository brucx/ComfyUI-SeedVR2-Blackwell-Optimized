# RTX Pro 6000 Blackwell 3B Optimization Notes

This branch now benchmarks the RTX Pro 6000 Blackwell path with the 3B model family only. The primary inference checkpoint is `seedvr2_ema_3b_fp8_e4m3fn.safetensors`.

## Fast preset

Use:

```bash
python inference_cli.py input.mov --blackwell_pro6000_preset --benchmark_json benchmark.json --debug
```

The preset now applies the fastest measured 3B FP8 path from the focused hypothesis sweep:

- `attention_mode=sdpa`
- no DiT or VAE `torch.compile`
- `seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- adaptive 4n+1 batch sizing capped at `batch_size=81`
- `uniform_batch_size=True`
- VAE Conv3D memory splitting disabled with `vae_conv_memory_limit_gb=0`

For the 81-frame 720p benchmark, this resolves to `batch_size=81 --uniform_batch_size --vae_conv_memory_limit_gb 0` and measured 28.2877s / 2.8634 FPS in `benchmark_results/blackwell_3b_fp8_preset_convlimit/`. The full clip measured 103.0863s / 2.9005 FPS in `benchmark_results/blackwell_3b_fp8_vae_decode_profile/full_convlimit_off/`. The previous SageAttention 3 + `torch.compile max-autotune` preset was slower on this 3B FP8 workload, so it remains available only through explicit flags.

A 300-frame SDPA sweep also supports the cap: batch 81 measured 117.0327s / 2.5548 FPS, while batch 149 regressed to 158.9516s / 1.8811 FPS.

VAE-only `torch.compile reduce-overhead` was tested as a follow-up because VAE encode/decode dominate the fast preset runtime. It regressed to 150.9339s / 0.5367 FPS and raised peak reserved VRAM to 36.41 GB, so the fast preset intentionally keeps VAE eager.

A full-clip VAE knob sweep also found no speed win from FP16 compute, disabling tensor offload, decode tiling, or encode+decode tiling. The current BF16 eager VAE path remains fastest at 116.1961s / 2.5732 FPS. Tiling is still useful as a memory knob: decode tiling reduced peak reserved VRAM from 20.22 GB to 17.44 GB, but slowed end-to-end time to 119.8899s.

Profiling `vae_decode` found the actionable VAE bottleneck: Conv3D memory splitting creates pad/cat/copy overhead and many small kernel launches. Disabling only Conv3D memory splitting with `--vae_conv_memory_limit_gb 0` improves full-clip throughput to 103.0863s / 2.9005 FPS, with peak reserved VRAM rising to 41.84 GB. Temporal causal slicing itself must stay enabled; disabling it OOMs at this resolution and batch size. Details are in `docs/blackwell_3b_fp8_vae_decode_profile.md`.

## Benchmark harness

Use:

```bash
python scripts/benchmark_blackwell.py --input /path/to/test.mov --output_dir benchmark_results/blackwell
```

The harness writes per-case logs, JSON benchmark records, `records.json`, and `summary.md`.

## ModelOpt NVFP4 W4A16 PTQ

Use:

```bash
python scripts/modelopt_nvfp4_ptq.py --model_dir ./models/SEEDVR2
python scripts/benchmark_blackwell.py --include_modelopt
```

The PTQ script uses NVIDIA Model Optimizer with NVFP4 weight quantizers and activation quantizers disabled, producing `seedvr2_ema_3b_nvfp4_w4a16.modelopt.pt`. The model loader restores `.modelopt.pt` files with `modelopt.torch.opt.restore`.

The conversion source defaults to `seedvr2_ema_3b_fp16.safetensors`. The 3B FP8 checkpoint is used for the main inference baseline and fast preset, but ModelOpt PTQ still needs the FP16 source because ModelOpt max calibration cannot directly reduce FP8 CUDA tensors in this environment.

`torch.compile` on the ModelOpt-quantized 3B DiT currently fails in TorchDynamo inside ModelOpt's dynamic callback wrapper. The benchmark harness records that failure, then runs the measured fallback with the ModelOpt DiT eager and VAE compiled with `max-autotune`.

## TRT subgraph + W4A4 QAT

`scripts/trt_w4a4_qat_probe.py` isolates the first 3B DiT video MLP branch, `dit.blocks[0].mlp.vid`, at the real 81-frame 720p token shape:

- boundary: `x[74655, 2560] -> y[74655, 2560]`
- parameters: `53,084,160`
- QAT: one SGD teacher-loss step
- working TRT path: ModelOpt ONNX deploy through `trtexec`

Full-shape probe result:

- FP16 eager: 36.2673ms
- FP16 Torch-TensorRT: 34.9448ms
- W4A4 eager: 38.4213ms
- W4A4 ModelOpt deploy/TRT profile: 21.4242ms
- W4A4 `DeviceModel` forward: 21.8382ms
- QAT loss: 0.020396
- max absolute error: 2.90625

## Integrated TRT MLP

The integrated inference path is enabled with:

```bash
python inference_cli.py input.mov --dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors --batch_size 81 --uniform_batch_size --attention_mode sageattn_3 --strict_attention_mode --trt_w4a4_mlp
```

It wraps `dit.blocks[0].mlp.vid`, uses the first live activation tensor to run one teacher-loss QAT step, compiles a fixed-shape W4A4 TensorRT `DeviceModel`, and returns that TensorRT output to the active 3B FP8 DiT block.

The 81-frame benchmark completed successfully in 73.4719s at 1.1025 FPS. Its W4A4 TRT profile latency was 20.3718ms for `x[74655, 2560]`, integrated compile time was 39.3830s, QAT loss was 0.001948, and max absolute error was 1.0 against the live teacher activation.

This is an integrated W4A4 TensorRT route for one fixed-shape DiT MLP subgraph, not a full-DiT TensorRT engine.
