# 3B FP8 VAE Optimization Probe

Input: requested `test.mov`, first 81 frames, 720p short-side output. Model: `seedvr2_ema_3b_fp8_e4m3fn.safetensors`.

| Case | Frames | Total sec | FPS | VAE encode | DiT upscale | VAE decode | Peak reserved VRAM | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fast preset reference | 81 | 31.2140 | 2.5950 | 7.4452s | 5.8466s | 15.3769s | 20.22 GB | reference |
| VAE compile reduce-overhead | 81 | 150.9339 | 0.5367 | 75.1262s | 5.9365s | 66.4138s | 36.41 GB | slower |

Conclusion: VAE-only `torch.compile` is not a useful default for the 3B FP8 Blackwell short-clip path. It increases both VAE phase time and peak reserved VRAM.
