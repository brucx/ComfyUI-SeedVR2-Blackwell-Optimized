# RTX Pro 6000 Blackwell Optimization Notes

This branch adds reproducible CLI controls and benchmark artifacts for the requested RTX Pro 6000 Blackwell path.

## Out-of-box preset

Use:

```bash
python inference_cli.py input.mov --blackwell_pro6000_preset --benchmark_json benchmark.json --debug
```

The preset applies:

- `attention_mode=sageattn_3` with `strict_attention_mode=True`
- DiT and VAE `torch.compile` using `backend=inductor`, `mode=max-autotune`
- `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors`
- `batch_size=81`
- `uniform_batch_size=True`

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

The PTQ script uses NVIDIA Model Optimizer with NVFP4 weight quantizers and activation quantizers disabled, giving a W4A16 checkpoint saved as `.modelopt.pt`. The model loader has been extended to restore `.modelopt.pt` files with `modelopt.torch.opt.restore`.

## TRT subgraph + W4A4 QAT status

TensorRT and ModelOpt are available in the target container, but the repository does not expose a static DiT subgraph boundary or a training/teacher-loss recipe for W4A4 QAT. The benchmark harness records this as an explicit incomplete experimental case instead of silently claiming a proxy result.

To complete this path, the next implementation step is a model-specific graph partition:

- isolate the MMDiT block forward signature after VAE latent preparation,
- export a stable ONNX or Torch-TensorRT subgraph at fixed `batch_size=81`,
- add ModelOpt NVFP4 activation quantizers for W4A4,
- run QAT against fp8/FP16 teacher outputs over representative video batches,
- compile the trained subgraph with TensorRT and route DiT block calls through the engine.
