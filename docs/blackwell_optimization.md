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
- compiles the same FP16 subgraph through ModelOpt ONNX deploy and `trtexec`,
- inserts ModelOpt NVFP4 W4A4 fake quantizers,
- runs a short teacher-loss QAT loop,
- benchmarks eager FP16, TRT FP16, and W4A4 eager,
- builds and profiles the W4A4 TensorRT path through ModelOpt ONNX deploy and `trtexec`,
- records the direct Torch-TensorRT frontend failures if Torch export cannot lower ModelOpt activation quantizers.

Current RTX Pro 6000 harness result for `seq_len=4096`: FP16 eager 2.3900ms, FP16 Torch-TensorRT 2.2937ms, FP16 ModelOpt deploy/TRT 2.55839ms, W4A4 eager 2.5223ms, W4A4 ModelOpt deploy/TRT profile 1.71290ms, and W4A4 ModelOpt `DeviceModel` forward 1.6587ms. The one-step QAT teacher loss was 2.156582 with SGD. The W4A4 `trtexec` profile reports 583.256 inferences/s for the isolated MLP subgraph, and the finite TensorRT output had max absolute error 50.75 against the FP16 teacher for the random 4096-token probe batch.

The same path also succeeds at `seq_len=74655`, which is the fixed token length for the requested 81-frame 720p benchmark after VAE latent shape `[21, 90, 158, 16]` and DiT patch size `[1, 2, 2]`. At this real full-shape boundary, FP16 eager averaged 48.8109ms, FP16 Torch-TensorRT averaged 44.5086ms, FP16 ModelOpt deploy/TRT reported 45.8629ms, W4A4 eager averaged 48.6992ms, W4A4 ModelOpt deploy/TRT reported 31.9918ms, and W4A4 ModelOpt `DeviceModel` forward averaged 30.9279ms over 5 iterations. The one-step QAT teacher loss was 2.134499 with SGD, and the finite W4A4 TensorRT output had max absolute error 71.0 against the FP16 teacher for the random full-shape probe batch. See `benchmark_results/blackwell_81f_720p_trt_qat_fullshape/trt_w4a4_qat_probe.json`.

The integrated inference path is enabled with `--trt_w4a4_mlp`. It wraps `dit.blocks[0].mlp.vid`, uses the first live activation tensor to run one teacher-loss QAT step, compiles a fixed-shape W4A4 TensorRT `DeviceModel`, and returns that TensorRT output to the active DiT block. The 81-frame benchmark using 7B FP16, SageAttention 3, batch 81, and the integrated MLP route completed successfully in 85.86s at 0.9434 FPS. Its W4A4 TRT profile latency was 30.5263ms for `x[74655, 3072]`, integrated compile time was 45.2335s, QAT loss was 0.005460, and max absolute error was 0.75 against the live teacher activation. See `benchmark_results/blackwell_81f_720p_trt_integrated_mlp/`.

Several ModelOpt 0.40 export workarounds are applied inside the probe:

- disable the empty FP8 ONNX exporter for this NVFP4 W4A4 path, because ModelOpt's FP8 detector also matches NVFP4 quantizer metadata;
- disable ONNX shape inference and optimization for this path, because the deploy stack can corrupt FP16 initializers in the quantized graph;
- restore original FP16 initializers from a pre-export state backup after NVFP4 post-processing;
- sanitize NVFP4 ONNX scale tensors to finite positive values before TensorRT parsing, because the default exporter can emit NaN/negative DQ scales for this subgraph.

The v5 and full-shape artifacts record no NaN/Inf initializers before or after ONNX save.

Direct Torch-TensorRT Dynamo export still fails on a fake tensor from `proj_in.input_quantizer.lifted_tensor_0`; the TorchScript frontend also fails because ModelOpt NVFP4 uses non-integer quantization without a `step_size`. The working W4A4 TRT path is therefore ModelOpt ONNX deploy through `trtexec`.

This is an integrated W4A4 TensorRT route for one fixed-shape DiT MLP subgraph, not a full-DiT TensorRT engine. The remaining work is scaling the same graph partitioning pattern to more DiT block components.

To complete this path, the next implementation step is a model-specific graph partition:

- isolate the MMDiT block forward signature after VAE latent preparation,
- export a stable ONNX or Torch-TensorRT subgraph at fixed `batch_size=81`,
- add ModelOpt NVFP4 activation quantizers for W4A4,
- run QAT against fp8/FP16 teacher outputs over representative video batches,
- compile the trained subgraph with TensorRT and route DiT block calls through the engine.
