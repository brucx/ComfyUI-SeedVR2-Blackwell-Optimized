# RTX Pro 6000 Blackwell 3B FP8 Benchmark Results

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM, CUDA 13.1, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`.

Input: requested `test.mov`, first 81 frames, 480x274 source, 720p short-side output (`1260x720`). The benchmark set below uses the 3B model family only; no 7B result is used for the current comparison.

| Case | Model / settings | Frames | Total sec | FPS | Peak reserved VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 3B FP8, SDPA, batch 5, no compile | 81 | 36.9087 | 2.1946 | 13.31 GB |
| 3B OOB cold | 3B FP8, SageAttention 3, `torch.compile max-autotune`, batch 81 | 81 | 264.5565 | 0.3062 | 35.68 GB |
| 3B OOB warm | Same as above, with Inductor cache reused | 81 | 166.3370 | 0.4870 | 34.28 GB |
| 3B ModelOpt fallback | 3B FP16 source -> NVFP4 W4A16 `.modelopt.pt`, SageAttention 3, DiT eager, VAE `torch.compile max-autotune`, batch 81 | 81 | 129.7714 | 0.6242 | 46.60 GB |
| 3B integrated W4A4 TRT MLP | 3B FP8, SageAttention 3, batch 81, first DiT video MLP routed through W4A4 TensorRT `DeviceModel` | 81 | 73.4719 | 1.1025 | 28.13 GB |
| 3B FP8 hypothesis winner | 3B FP8, SDPA, batch 81, uniform batch, no compile | 81 | 31.2787 | 2.5896 | 20.22 GB |
| 3B FP8 fast preset | `--blackwell_pro6000_preset`: 3B FP8, SDPA, adaptive batch 81, uniform batch, no compile | 81 | 31.2140 | 2.5950 | 20.22 GB |
| 3B FP8 VAE Conv3D unsplit preset | 3B FP8, SDPA, batch 81, uniform batch, no compile, `--vae_conv_memory_limit_gb 0` | 81 | 28.2877 | 2.8634 | 40.97 GB |
| 3B FP8 1080p preset | 3B FP8, SDPA, batch 81, uniform batch, no compile, `--vae_conv_memory_limit_gb 0`, 1890x1080 output | 81 | 80.0533 | 1.0118 | 94.16 GB |
| 3B FP8 1080p full clip | Same 1080p preset on full `test.mov` | 299 | 241.7233 | 1.2370 | 93.16 GB |

Phase timings:

| Case | VAE encode | DiT upscale | VAE decode | Postprocess |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 7.6326s | 10.5793s | 15.5863s | 1.8353s |
| 3B OOB cold | 61.6734s | 147.4916s | 52.9900s | 1.1794s |
| 3B OOB warm | 58.3886s | 49.2426s | 56.1972s | 1.2732s |
| 3B ModelOpt fallback | 56.5573s | 17.7272s | 53.1894s | 1.1771s |
| 3B integrated W4A4 TRT MLP | 7.5292s | 47.5204s | 15.7619s | 1.2446s |

Artifacts:

- Baseline and cold OOB: `benchmark_results/blackwell_3b_fp8_81f_720p/`
- Warm OOB: `benchmark_results/blackwell_3b_fp8_81f_720p_warm_oob/`
- ModelOpt W4A16: `benchmark_results/blackwell_3b_fp8_81f_720p_modelopt/`
- W4A4 TRT/QAT and integrated inference: `benchmark_results/blackwell_3b_fp8_81f_720p_trt/`
- 3B FP8 optimization hypotheses: `benchmark_results/blackwell_3b_fp8_hypotheses/`
- 3B FP8 fast preset validation: `benchmark_results/blackwell_3b_fp8_preset_fast/`
- 3B FP8 300-frame SDPA batch sweep: `benchmark_results/blackwell_3b_fp8_batch_sweep_300f/`
- 3B FP8 VAE compile probe: `benchmark_results/blackwell_3b_fp8_vae_probe/`
- 3B FP8 full-clip VAE acceleration plan: `benchmark_results/blackwell_3b_fp8_vae_plan_full/`
- 3B FP8 VAE decode profile and Conv3D memory split validation: `benchmark_results/blackwell_3b_fp8_vae_decode_profile/`
- 3B FP8 Conv3D-unsplit preset validation: `benchmark_results/blackwell_3b_fp8_preset_convlimit/`
- 3B FP8 1080p preset validation: `benchmark_results/blackwell_3b_fp8_1080p_preset/`
- 3B FP8 1080p full clip validation: `benchmark_results/blackwell_3b_fp8_1080p_preset_full/`

Engineering stage status:

- ModelOpt NVFP4 W4A16 PTQ loader support remains `.modelopt.pt` through `modelopt.torch.opt.restore`.
- The current 3B ModelOpt checkpoint is `seedvr2_ema_3b_nvfp4_w4a16.modelopt.pt`, generated from `seedvr2_ema_3b_fp16.safetensors`. The 3B FP8 checkpoint is the requested inference baseline, but ModelOpt PTQ still needs the FP16 source because ModelOpt max calibration does not support reducing FP8 CUDA tensors directly.
- `torch.compile` on the ModelOpt-quantized 3B DiT still fails inside ModelOpt's dynamic callback wrapper. The measured engineering fallback keeps the ModelOpt DiT eager and compiles only the VAE.

Extreme stage status:

- The full-shape 3B W4A4 TRT probe targets `dit.blocks[0].mlp.vid`, boundary `x[74655, 2560] -> y[74655, 2560]`, with 53,084,160 parameters.
- Probe latency at the full 81-frame token shape: FP16 eager 36.2673ms, FP16 TRT 34.9448ms, W4A4 eager 38.4213ms, W4A4 ModelOpt deploy/TRT 21.4242ms, and W4A4 `DeviceModel` forward 21.8382ms over 20 iterations. One-step QAT loss was 0.020396; max absolute error was 2.90625.
- Integrated 3B FP8 inference uses `--trt_w4a4_mlp` and routes the first live video MLP activation through the W4A4 TensorRT `DeviceModel`. Compile metadata: W4A4 TRT profile 20.3718ms, integrated compile time 39.3830s, one-step QAT loss 0.001948, and max absolute error 1.0 against the live teacher activation.

Conclusion:

- Under the requested 3B FP8 comparison, the fastest measured end-to-end path for this short 81-frame clip is now the simple SDPA batch-size optimization: batch 81 with uniform batching, no SageAttention, and no compile. In the focused hypothesis run it measured 31.2787s / 2.5896 FPS, 1.096x faster than the same-run batch 5 reference.
- The previous integrated W4A4 TRT MLP route improves substantially over the 3B OOB warm compile path (166.3370s / 0.4870 FPS) and the 3B ModelOpt fallback (129.7714s / 0.6242 FPS), but it does not beat the lean 3B FP8 SDPA path.
- Focused hypothesis testing found that SageAttention 3, DiT-only `torch.compile`, and disabling tensor offload do not improve the 3B FP8 short-clip path. Details are in `docs/blackwell_3b_fp8_hypotheses.md`.
- `--blackwell_pro6000_preset` now maps to the measured fast path: 3B FP8, SDPA, no compile, adaptive 4n+1 batch sizing capped at 81, uniform batches, and VAE Conv3D memory splitting disabled. The validation run measured 28.2877s / 2.8634 FPS.
- A longer 300-frame SDPA sweep confirms the 81 cap: batch 81 measured 117.0327s / 2.5548 FPS, while batch 149 regressed to 158.9516s / 1.8811 FPS.
- VAE-only `torch.compile reduce-overhead` is not a follow-up win for the fast path: it measured 150.9339s / 0.5367 FPS and raised peak reserved VRAM to 36.41 GB.
- Full-clip VAE acceleration testing found no speed win from FP16 compute, disabling tensor offload, decode tiling, or encode+decode tiling. The best measured VAE path remains the current BF16 eager VAE with CPU tensor offload: 116.1961s / 2.5732 FPS on the full clip.
- `vae_decode` profiling found that Conv3D memory splitting creates substantial pad/cat/copy overhead. Disabling only that split with `--vae_conv_memory_limit_gb 0` improved the full clip from 116.1693s / 2.5738 FPS to 103.0863s / 2.9005 FPS, with peak reserved VRAM rising from 20.22 GB to 41.84 GB. Temporal causal slicing itself cannot be disabled; it OOMs on the 81-frame 720p batch.
- The same fast preset can upscale to 1080p output (1890x1080) on the 95GB Blackwell card. Full `test.mov` measured 241.7233s / 1.2370 FPS, but peak reserved VRAM reached 93.16 GB, so this setting is very close to the device limit.
