# 3B FP8 Optimization Hypothesis Benchmarks

| case | hypothesis | rc | frames | seconds | fps | phase2 | peak reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h0_baseline_sdpa_b5 | reference: current 3B FP8 baseline | 0 | 81 | 34.266 | 2.3639 | 8.6283 | 13.3125 |
| h1_sdpa_b21 | moderate batch size may reduce scheduling overhead without hurting VAE | 0 | 81 | 32.9432 | 2.4588 | 6.68 | 20.0 |
| h2_sdpa_b81 | large batch 81 may improve DiT throughput without attention/compile changes | 0 | 81 | 31.2787 | 2.5896 | 5.909 | 20.21875 |
| h3_sage3_b5 | SageAttention 3 alone may improve attention at baseline batch size | 0 | 81 | 34.9537 | 2.3174 | 8.8729 | 13.3125 |
| h4_sage3_b21 | SageAttention 3 with moderate batch may be better balanced | 0 | 81 | 33.5981 | 2.4109 | 7.0878 | 20.0 |
| h5_sage3_b81 | SageAttention 3 with batch 81 isolates attention+batch without compile | 0 | 81 | 31.7975 | 2.5474 | 6.3634 | 20.21875 |
| h6_sage3_b81_compile_dit | compile DiT only may help while avoiding VAE compile regression | 0 | 81 | 67.3036 | 1.2035 | 41.8172 | 33.3125 |
| h7_sdpa_b5_no_tensor_offload | keeping intermediate tensors on GPU may avoid CPU transfer overhead | 0 | 81 | 35.0195 | 2.313 | 8.8913 | 14.34375 |
