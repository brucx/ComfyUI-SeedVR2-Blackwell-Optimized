# RTX Pro 6000 Blackwell Benchmark Results

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM, CUDA 13.1, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`.

Input: `test.mov` from the requested URL, first 81 frames, 480x274 source, 720p short-side output (`1260x720`). This matches the requested `batch_size=81` path.

| Case | Model / settings | Frames | Total sec | FPS | Peak reserved VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 3B FP8, SDPA, batch 5, no compile | 81 | 73.346 | 1.1044 | 13.31 GB |
| Blackwell preset cold | 7B FP8 mixed block35, SageAttention 3, `torch.compile max-autotune`, batch 81 | 81 | 501.3785 | 0.1616 | 47.41 GB |
| Blackwell preset warm | Same as above, with Inductor cache reused | 81 | 178.3763 | 0.4541 | 39.97 GB |

Phase timings:

| Case | VAE encode | DiT upscale | VAE decode | Postprocess |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 7.0155s | 9.5952s | 15.6733s | 3.1735s |
| Blackwell preset cold | 68.8646s | 247.2887s | 182.5027s | 1.2432s |
| Blackwell preset warm | 62.6533s | 55.8378s | 57.1879s | 1.2237s |

Compatibility fixes made during benchmarking:

- Disabled the default `cudaMallocAsync` allocator when `torch.compile` or the Blackwell preset is requested, because PyTorch cudagraph pool checks failed with `cudaMallocAsync`.
- Added `torch.compiler.cudagraph_mark_step_begin()` before repeated compiled VAE/DiT invocations to avoid cudagraph output overwrite errors.
- Built and installed official `sageattention3_blackwell` from `thu-ml/SageAttention` for sm_120a; strict `sageattn_3` validation passes.

Engineering stage status:

- ModelOpt NVFP4 W4A16 PTQ loader support was added for `.modelopt.pt` through `modelopt.torch.opt.restore`.
- The PTQ creation attempt on this container segfaulted inside CUDA allocation during ModelOpt initialization; see `benchmark_results/modelopt_nvfp4/ptq.log`.
- TRT subgraph + W4A4 QAT is documented as not completed because this repository lacks a stable exported DiT subgraph boundary and QAT training recipe.
