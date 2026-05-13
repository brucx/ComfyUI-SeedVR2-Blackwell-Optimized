# RTX Pro 6000 Blackwell Benchmark Results

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM, CUDA 13.1, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`.

Input: `test.mov` from the requested URL, first 81 frames, 480x274 source, 720p short-side output (`1260x720`). This matches the requested `batch_size=81` path.

| Case | Model / settings | Frames | Total sec | FPS | Peak reserved VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 3B FP8, SDPA, batch 5, no compile | 81 | 73.346 | 1.1044 | 13.31 GB |
| Blackwell preset cold | 7B FP8 mixed block35, SageAttention 3, `torch.compile max-autotune`, batch 81 | 81 | 501.3785 | 0.1616 | 47.41 GB |
| Blackwell preset warm | Same as above, with Inductor cache reused | 81 | 178.3763 | 0.4541 | 39.97 GB |
| ModelOpt NVFP4 fallback | 7B FP16 base -> NVFP4 W4A16 `.modelopt.pt`, SageAttention 3, DiT eager, VAE `torch.compile max-autotune`, batch 81 | 81 | 158.6656 | 0.5105 | 66.44 GB |
| Integrated W4A4 TRT MLP | 7B FP16, SageAttention 3, batch 81, first DiT video MLP routed through W4A4 TensorRT `DeviceModel` | 81 | 85.8600 | 0.9434 | 35.44 GB |

Phase timings:

| Case | VAE encode | DiT upscale | VAE decode | Postprocess |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 7.0155s | 9.5952s | 15.6733s | 3.1735s |
| Blackwell preset cold | 68.8646s | 247.2887s | 182.5027s | 1.2432s |
| Blackwell preset warm | 62.6533s | 55.8378s | 57.1879s | 1.2237s |
| ModelOpt NVFP4 fallback | 60.7157s | 34.7902s | 60.8164s | 1.2458s |
| Integrated W4A4 TRT MLP | 7.3964s | 60.5726s | 15.6004s | 1.2242s |

Compatibility fixes made during benchmarking:

- Disabled the default `cudaMallocAsync` allocator when `torch.compile` or the Blackwell preset is requested, because PyTorch cudagraph pool checks failed with `cudaMallocAsync`.
- Added `torch.compiler.cudagraph_mark_step_begin()` before repeated compiled VAE/DiT invocations to avoid cudagraph output overwrite errors.
- Built and installed official `sageattention3_blackwell` from `thu-ml/SageAttention` for sm_120a; strict `sageattn_3` validation passes.

Engineering stage status:

- ModelOpt NVFP4 W4A16 PTQ loader support was added for `.modelopt.pt` through `modelopt.torch.opt.restore`.
- ModelOpt PTQ now succeeds from the 7B FP16 checkpoint; see `benchmark_results/modelopt_nvfp4/ptq_unbuffered4_fp16base.log`.
- The already-FP8 mixed block35 checkpoint cannot be used as the PTQ source because ModelOpt's max calibration calls `torch.max` on FP8 weights, and PyTorch reports `max_all_cuda` is not implemented for `Float8_e4m3fn`.
- `torch.compile` on the ModelOpt-quantized DiT currently fails in TorchDynamo inside ModelOpt's dynamic callback wrapper; see `benchmark_results/blackwell_81f_720p_modelopt/modelopt_nvfp4_w4a16.log`.
- The measured engineering fallback keeps the ModelOpt DiT eager and compiles only the VAE; DiT inference for the single 81-frame batch was 6.8964s, while VAE encode/decode dominated runtime.

Extreme stage status:

- Added `scripts/trt_w4a4_qat_probe.py` to isolate a real weighted DiT subgraph: `dit.blocks[0].mlp.vid`, with boundary `x[seq_len, 3072] -> y[seq_len, 3072]` and 75,512,832 parameters.
- On `seq_len=4096`, FP16 Torch-TensorRT compilation of that subgraph succeeds: eager FP16 averaged 2.3900ms, TRT FP16 averaged 2.2937ms over 20 iterations; see `benchmark_results/blackwell_81f_720p_trt_qat_harness_v5/trt_w4a4_qat_probe.json`.
- ModelOpt deploy also builds and profiles an FP16 TensorRT engine through ONNX/trtexec: 2.55839ms reported latency for the same subgraph.
- ModelOpt NVFP4 W4A4 fake-quant insertion succeeds on the same subgraph, and one teacher-loss QAT step runs with SGD and loss 2.156582; W4A4 eager averaged 2.5223ms over 20 iterations.
- W4A4 ModelOpt deploy now exports a finite NVFP4 ONNX graph, builds/profiles a TensorRT engine through `trtexec`, and runs `DeviceModel` forward: 1.71290ms reported `trtexec` latency, 583.256 inferences/s, and 1.6587ms average CUDA-event forward time over 20 iterations. The finite W4A4 TRT output had max absolute error 50.75 against the FP16 teacher for this random 4096-token probe batch.
- The same probe also succeeds at the real 81-frame 720p token length, `seq_len=74655`, derived from encoded latent shape `[21, 90, 158, 16]` and DiT patch size `[1, 2, 2]`. At this fixed full-shape boundary, FP16 eager averaged 48.8109ms, FP16 Torch-TensorRT averaged 44.5086ms, FP16 ModelOpt deploy/TRT reported 45.8629ms, W4A4 eager averaged 48.6992ms, W4A4 ModelOpt deploy/TRT reported 31.9918ms, and W4A4 `DeviceModel` forward averaged 30.9279ms over 5 iterations. One-step QAT loss was 2.134499 with SGD, and finite W4A4 TRT output had max absolute error 71.0 against the FP16 teacher for the random full-shape probe batch. See `benchmark_results/blackwell_81f_720p_trt_qat_fullshape/trt_w4a4_qat_probe.json`.
- Added an integrated fixed-shape path that wraps `dit.blocks[0].mlp.vid` and routes the live first-block MLP activation through the W4A4 TensorRT `DeviceModel` during actual inference. The 81-frame run succeeded with 85.86s total time, 0.9434 FPS, 35.44GB peak reserved VRAM, 45.2335s integrated compile time, W4A4 TRT profile latency 30.5263ms, one-step QAT loss 0.005460, and max absolute error 0.75 against the live FP16 teacher activation. See `benchmark_results/blackwell_81f_720p_trt_integrated_mlp/`.
- The W4A4 export path requires local ModelOpt 0.40 workarounds in the probe: bypass the empty FP8 exporter that also matches NVFP4 quantizers, disable ONNX shape inference/optimization for this path, restore original FP16 initializers from a pre-export state backup, and sanitize NVFP4 scale tensors before TensorRT parsing. The v5 artifact records no NaN/Inf initializers before or after ONNX save.
- Torch-TensorRT Dynamo and TorchScript frontends still fail directly on the ModelOpt fake-quantized module (`proj_in.input_quantizer.lifted_tensor_0` fake tensor and NVFP4 non-integer `step_size`), so the working W4A4 TRT path is ModelOpt ONNX deploy through `trtexec`.
- This is an integrated W4A4 TensorRT route for one fixed-shape DiT MLP subgraph, not a full-DiT TensorRT engine.
