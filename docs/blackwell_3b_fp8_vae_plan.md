# Blackwell 3B FP8 VAE Acceleration Test Plan

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM.

Input: full requested `test.mov`, 299 processed frames, 480x274 source, 720p short-side output (`1260x720`). All cases use `seedvr2_ema_3b_fp8_e4m3fn.safetensors`, SDPA attention, batch 81, uniform batches, and eager DiT.

Artifacts: `benchmark_results/blackwell_3b_fp8_vae_plan_full/`

## Plan

| Case | Hypothesis | Change |
| --- | --- | --- |
| baseline_bf16 | Current fast path is the reference | bf16 compute, CPU tensor offload, eager VAE |
| fp16_compute | FP16 may use faster VAE Conv3D kernels than BF16 | `--compute_dtype fp16` |
| no_tensor_offload | Keeping latents on GPU may avoid CPU transfer overhead | `--tensor_offload_device none` |
| decode_tiled | Decode tiling may improve locality despite extra tile overhead | `--vae_decode_tiled` |
| encode_decode_tiled | Tiling both VAE phases may reduce memory pressure and improve locality | `--vae_encode_tiled --vae_decode_tiled` |

`--compile_vae --compile_mode reduce-overhead` was already tested separately on the 81-frame fast path and is included as prior evidence: 150.9339s / 0.5367 FPS, with VAE encode/decode rising to 75.1262s / 66.4138s and peak reserved VRAM rising to 36.41 GB.

## Full-Clip Results

| Case | Total sec | FPS | Speed vs baseline | VAE encode | DiT upscale | VAE decode | Postprocess | Peak reserved VRAM | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_bf16 | 116.1961 | 2.5732 | 1.000x | 27.0976s | 18.4924s | 61.6322s | 3.7200s | 20.22 GB | fastest |
| fp16_compute | 122.3890 | 2.4430 | 0.949x | 28.0341s | 19.0975s | 63.1983s | 3.7841s | 20.19 GB | slower |
| no_tensor_offload | 118.3482 | 2.5264 | 0.982x | 27.0740s | 19.1902s | 62.4929s | 4.7179s | 20.56 GB | slower |
| decode_tiled | 119.8899 | 2.4940 | 0.969x | 27.6129s | 18.7121s | 66.1108s | 3.7153s | 17.44 GB | slower, lower VRAM |
| encode_decode_tiled | 121.1064 | 2.4689 | 0.959x | 28.8906s | 18.6734s | 66.0785s | 3.6801s | 17.44 GB | slower, lower VRAM |

## Conclusion

None of the low-risk VAE knobs improves full-clip inference speed. The current fast preset should keep:

- `compute_dtype=auto`, which resolves to BF16 on this Blackwell host
- `tensor_offload_device=cpu`
- VAE tiling disabled
- VAE `torch.compile` disabled

The only measured VAE tradeoff worth keeping is tiling for memory reduction. Decode tiling lowers peak reserved VRAM from 20.22 GB to 17.44 GB, but costs about 3.2% end-to-end speed. It should remain an opt-in memory knob, not a performance default.

The next VAE acceleration path should be internal profiling of `vae_decode`, especially Conv3D/causal slicing kernels. Whole-module dtype, tiling, tensor placement, and `torch.compile` did not produce a speed win.
