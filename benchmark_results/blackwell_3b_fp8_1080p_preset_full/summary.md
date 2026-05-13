# 3B FP8 1080p Full-Clip Preset Benchmark

Input: full requested `test.mov`, 299 processed frames, 480x274 source, 1080p short-side output (`1890x1080`).

Settings: `--blackwell_pro6000_preset`, which resolves to 3B FP8, SDPA, batch 81, uniform batches, eager DiT/VAE, and `--vae_conv_memory_limit_gb 0`.

| Case | Frames | Seconds | FPS | VAE encode | DiT upscale | VAE decode | Postprocess | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preset_1080p_full | 299 | 241.7233 | 1.2370 | 56.5059s | 41.2521s | 128.0804s | 9.3369s | 93.16 GB |

Conclusion: Full-clip 1080p succeeds on the RTX Pro 6000 Blackwell 95GB card. Throughput is 1.2370 FPS, and peak reserved VRAM is 93.16 GB.
