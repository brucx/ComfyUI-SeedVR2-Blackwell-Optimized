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

The conversion source defaults to `seedvr2_ema_7b_fp16.safetensors`. The already-FP8 mixed block35 checkpoint is not a viable PTQ source in this environment because ModelOpt's max calibration calls `torch.max` on FP8 weights, and PyTorch does not implement that CUDA reduction for `Float8_e4m3fn`.

The compiled ModelOpt DiT path currently fails in TorchDynamo inside ModelOpt's dynamic callback wrapper. The benchmark harness records that failure, then runs a fallback engineering case with the ModelOpt DiT eager and the VAE still compiled with `max-autotune`.

## TRT subgraph + W4A4 QAT status

TensorRT and ModelOpt are available in the target container. This branch adds `scripts/trt_w4a4_qat_probe.py` as a concrete first extreme-stage probe against a real weighted DiT subgraph: `dit.blocks[0].mlp.vid`.

The probe:

- loads real 7B FP16 DiT weights,
- extracts the first block's video MLP branch with boundary `x[seq_len, 3072] -> y[seq_len, 3072]`,
- compiles that subgraph with FP16 Torch-TensorRT,
- inserts ModelOpt NVFP4 W4A4 fake quantizers,
- runs a short teacher-loss QAT loop,
- benchmarks eager FP16, TRT FP16, and W4A4 eager,
- records the W4A4 TensorRT export failure if Torch export cannot lower ModelOpt activation quantizers.

Current RTX Pro 6000 harness result for `seq_len=4096`: FP16 eager 2.3871ms, FP16 TRT 2.2786ms, W4A4 eager 2.1829ms. W4A4 TRT export fails in the Dynamo frontend because Torch export sees a fake tensor from `proj_in.input_quantizer.lifted_tensor_0`; the TorchScript frontend also fails because ModelOpt NVFP4 uses non-integer quantization without a `step_size`.

This is not yet a full-pipeline TRT W4A4 engine. The remaining work is replacing the ModelOpt fake-quant activation boundary with a TensorRT-exportable quantize/dequantize representation and routing the DiT block calls through the compiled engine.

To complete this path, the next implementation step is a model-specific graph partition:

- isolate the MMDiT block forward signature after VAE latent preparation,
- export a stable ONNX or Torch-TensorRT subgraph at fixed `batch_size=81`,
- add ModelOpt NVFP4 activation quantizers for W4A4,
- run QAT against fp8/FP16 teacher outputs over representative video batches,
- compile the trained subgraph with TensorRT and route DiT block calls through the engine.
