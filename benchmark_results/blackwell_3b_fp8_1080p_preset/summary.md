# 3B FP8 1080p Preset Benchmark

Input: requested `test.mov`, first 81 frames, 480x274 source, 1080p short-side output (`1890x1080`).

Settings: `--blackwell_pro6000_preset`, which resolves to 3B FP8, SDPA, batch 81, uniform batches, eager DiT/VAE, and `--vae_conv_memory_limit_gb 0`.

| Case | Frames | Seconds | FPS | VAE encode | DiT upscale | VAE decode | Postprocess | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preset_1080p_81f | 81 | 80.0533 | 1.0118 | 14.4558s | 11.3082s | 49.6005s | 2.4648s | 94.16 GB |

Conclusion: 1080p works for 81 frames, but peak reserved VRAM is extremely close to the 94.97 GB device limit.
