# Blackwell 3B FP8 Inference Speed Hypotheses

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95GB VRAM.

Input: requested `test.mov`, first 81 frames, 480x274 source, 720p short-side output (`1260x720`). All cases use `seedvr2_ema_3b_fp8_e4m3fn.safetensors`; no 7B model is used.

Artifacts: `benchmark_results/blackwell_3b_fp8_hypotheses/`

## Results

### 81-frame hypothesis sweep

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

### 300-frame SDPA batch sweep

Artifacts: `benchmark_results/blackwell_3b_fp8_batch_sweep_300f/`

| Case | Frames | Total sec | FPS | Speed vs batch 5 | DiT phase | Peak reserved VRAM | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SDPA batch 5 | 299 | 127.6574 | 2.3422 | 1.000x | 26.8458s | 13.31 GB | reference |
| SDPA batch 21 | 299 | 119.7791 | 2.4963 | 1.066x | 20.8266s | 20.00 GB | faster |
| SDPA batch 81 | 299 | 117.0327 | 2.5548 | 1.091x | 18.6395s | 20.22 GB | fastest |
| SDPA batch 149 | 299 | 158.9516 | 1.8811 | 0.803x | 25.1271s | 28.38 GB | slower |

## Interpretation

The winning path is not a new kernel or quantization route. For this 81-frame 3B FP8 workload, the main bottleneck exposed by the hypothesis tests is per-batch overhead in the DiT phase. Increasing `--batch_size` and using `--uniform_batch_size` reduces the number of DiT scheduling launches enough to improve end-to-end time, while the extra VRAM stays modest on the 95GB Blackwell card.

The longer 300-frame sweep confirms the same direction but also validates the preset cap. Batch 81 remains the best measured setting. Batch 149 increases padding/working-set cost enough to lose to batch 81 despite using fewer logical batches, so raising the Blackwell preset cap above 81 is not justified by this clip.

SageAttention 3 does not win in this shape. It is neutral to slightly negative compared with SDPA at the same batch size, so attention kernel replacement is not the best next step unless a different token shape or longer clip changes the balance.

`torch.compile` is counterproductive for this short clip. Even when restricted to the DiT and with VAE compile avoided, compile/runtime overhead dominates and makes the case about 2x slower than the SDPA batch 81 run.

VAE-only `torch.compile reduce-overhead` is also counterproductive on the fast 3B FP8 path. The 81-frame probe measured 150.9339s / 0.5367 FPS, with VAE encode/decode rising to 75.1262s / 66.4138s and peak reserved VRAM rising to 36.41 GB. The artifact is `benchmark_results/blackwell_3b_fp8_vae_probe/`.

Keeping tensors on GPU by setting `--tensor_offload_device none` also does not help here. It raises peak reserved VRAM slightly and slows the run, so the default offload behavior is not the limiting factor for this clip.

## Recommended Next Optimization Paths

1. Promote an adaptive batch-size heuristic for 3B FP8 Blackwell inference. Implemented in `--blackwell_pro6000_preset`; for this benchmark, it resolves to `--batch_size 81 --uniform_batch_size`.
2. Test the same SDPA batch-size sweep on longer clips. Completed on the 300-frame source clip; batch 81 still wins, and batch 149 regresses.
3. Investigate VAE encode/decode separately. VAE-only compile was tested and regressed badly; future VAE work should focus on kernel-level or tiling/data-movement changes, not enabling `torch.compile` by default.
4. Deprioritize SageAttention 3 and `torch.compile` for the 3B FP8 short-clip path unless a longer-clip benchmark shows a different tradeoff.
