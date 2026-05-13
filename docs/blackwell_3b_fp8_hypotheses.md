# Blackwell 3B FP8 Inference Speed Hypotheses

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM.

Input: requested `test.mov`, first 81 frames, 480x274 source, 720p short-side output (`1260x720`). All cases use `seedvr2_ema_3b_fp8_e4m3fn.safetensors`; no 7B model is used.

Artifacts: `benchmark_results/blackwell_3b_fp8_hypotheses/`

## Results

| Case | Hypothesis | Total sec | FPS | Speed vs h0 | Peak reserved VRAM | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| h0 | Current 3B FP8 SDPA batch 5 baseline | 34.2660 | 2.3639 | 1.000x | 13.31 GB | reference |
| h1 | SDPA batch 21 reduces DiT scheduling overhead | 32.9432 | 2.4588 | 1.040x | 20.00 GB | faster |
| h2 | SDPA batch 81 maximizes DiT throughput without compile overhead | 31.2787 | 2.5896 | 1.096x | 20.22 GB | fastest |
| h3 | SageAttention 3 alone improves attention kernels | 34.9537 | 2.3174 | 0.980x | 13.31 GB | slower |
| h4 | SageAttention 3 plus batch 21 balances attention and batching | 33.5981 | 2.4109 | 1.020x | 20.00 GB | slightly faster than h0, slower than SDPA batch 21 |
| h5 | SageAttention 3 plus batch 81 improves on SDPA batch 81 | 31.7975 | 2.5474 | 1.078x | 20.22 GB | faster than h0, slower than SDPA batch 81 |
| h6 | DiT compile only helps when VAE compile is avoided | 67.3036 | 1.2035 | 0.509x | 33.31 GB | much slower |
| h7 | Disabling tensor offload avoids transfer overhead | 35.0195 | 2.3130 | 0.978x | 14.34 GB | slower |

## Interpretation

The winning path is not a new kernel or quantization route. For this 81-frame 3B FP8 workload, the main bottleneck exposed by the hypothesis tests is per-batch overhead in the DiT phase. Increasing `--batch_size` and using `--uniform_batch_size` reduces the number of DiT scheduling launches enough to improve end-to-end time, while the extra VRAM stays modest on the 95GB Blackwell card.

SageAttention 3 does not win in this shape. It is neutral to slightly negative compared with SDPA at the same batch size, so attention kernel replacement is not the best next step unless a different token shape or longer clip changes the balance.

`torch.compile` is counterproductive for this short clip. Even when restricted to the DiT and with VAE compile avoided, compile/runtime overhead dominates and makes the case about 2x slower than the SDPA batch 81 run.

Keeping tensors on GPU by setting `--tensor_offload_device none` also does not help here. It raises peak reserved VRAM slightly and slows the run, so the default offload behavior is not the limiting factor for this clip.

## Recommended Next Optimization Paths

1. Promote an adaptive batch-size heuristic for 3B FP8 Blackwell inference. Implemented in `--blackwell_pro6000_preset`; for this benchmark, it resolves to `--batch_size 81 --uniform_batch_size`.
2. Test the same SDPA batch-size sweep on longer clips. If the benefit scales, this is the lowest-risk production optimization.
3. Investigate VAE encode/decode separately. Once DiT batching is improved, VAE work becomes a larger share of total runtime.
4. Deprioritize SageAttention 3 and `torch.compile` for the 3B FP8 short-clip path unless a longer-clip benchmark shows a different tradeoff.
