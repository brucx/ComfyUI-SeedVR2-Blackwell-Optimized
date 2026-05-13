# Blackwell 3B FP8 VAE Decode Profile

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM.

Input: requested `test.mov`, 720p short-side output. All inference cases use `seedvr2_ema_3b_fp8_e4m3fn.safetensors`, SDPA attention, batch 81, uniform batches, eager DiT, and BF16 compute.

Artifacts: `benchmark_results/blackwell_3b_fp8_vae_decode_profile/`

## Profile Findings

The default VAE decode path is dominated by repeated causal Conv3D and normalization work caused by temporal causal slicing plus Conv3D memory splitting. On the first 81-frame decode batch, the torch profiler recorded:

| Category from top profiler ops | Default total CUDA time in top ops | Default calls | Conv limit disabled total CUDA time in top ops | Conv limit disabled calls |
| --- | ---: | ---: | ---: | ---: |
| launch / command buffer overhead | 63631.694 ms | 33592 | 52169.752 ms | 16307 |
| cuDNN convolution | 29633.965 ms | 1273 | 28224.739 ms | 779 |
| group norm | 19637.017 ms | 2729 | 16295.183 ms | 2777 |
| copy / contiguous / layout conversion | 5891.400 ms | 14236 | 3849.372 ms | 6398 |
| cat | 4295.260 ms | 2627 | 2268.851 ms | 1027 |
| pad | 2595.065 ms | 1596 | 0.000 ms | 0 |

Profiler instrumentation adds large overhead, so these numbers are for attribution, not end-to-end FPS. The non-profiled validation runs below are the speed reference.

## Validation Runs

| Case | Frames | Total sec | FPS | VAE encode | VAE decode phase | Inner `vae_decode` timer | Peak reserved VRAM | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 81-frame fast preset reference | 81 | 31.2140 | 2.5950 | 7.4452s | 15.3769s | 14.1844s | 20.22 GB | reference |
| 81-frame `--vae_conv_memory_limit_gb 0` | 81 | 28.6162 | 2.8306 | 6.5851s | 13.5041s | 11.6760s | 40.97 GB | faster |
| 81-frame updated preset validation | 81 | 28.2877 | 2.8634 | 6.5334s | 13.3488s | 11.5937s | 40.97 GB | faster |
| Full clip fast preset reference | 299 | 116.1693 | 2.5738 | 26.9956s | 61.3540s | 14.5095s per logged batch | 20.22 GB | reference |
| Full clip `--vae_conv_memory_limit_gb 0` | 299 | 103.0863 | 2.9005 | 23.8534s | 53.5754s | 11.9642s per logged batch | 41.84 GB | faster |

Temporal causal slicing was also tested with `--vae_causal_slice_size 0`. It is not viable at this resolution and batch size: encode OOMed with 71.25 GiB already allocated and a 17.58 GiB request. Combining temporal slicing disabled with Conv3D splitting disabled also OOMed. Temporal slicing must stay enabled.

## Conclusion

The actionable VAE decode bottleneck is the Conv3D memory splitting path, not BF16, tensor offload, tiling, or whole-module compile.

Disabling `InflatedCausalConv3d` memory splitting with `--vae_conv_memory_limit_gb 0` removes pad overhead, cuts cat/copy calls, reduces cuDNN convolution call count, and improves full-clip FPS from 2.5738 to 2.9005. The tradeoff is higher peak reserved VRAM: about 41.84 GB instead of 20.22 GB.

For RTX Pro 6000 Blackwell 95GB, this is a good high-throughput default. For lower VRAM cards, keep the memory limit enabled.
