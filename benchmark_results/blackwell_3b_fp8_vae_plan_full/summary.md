# 3B FP8 Full-Clip VAE Acceleration Plan

| case | hypothesis | rc | frames | seconds | fps | vae encode | vae decode | peak reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_bf16 | current fast path: bf16 compute, CPU tensor offload, eager VAE | 0 | 299 | 116.1961 | 2.5732 | 27.0976 | 61.6322 | 20.21875 |
| fp16_compute | fp16 compute may use faster VAE Conv3D kernels than bf16 | 0 | 299 | 122.389 | 2.443 | 28.0341 | 63.1983 | 20.1875 |
| no_tensor_offload | keeping latents on GPU may reduce CPU transfer overhead between VAE and DiT | 0 | 299 | 118.3482 | 2.5264 | 27.074 | 62.4929 | 20.5625 |
| decode_tiled | VAE decode tiling may improve locality despite extra tile overhead | 0 | 299 | 119.8899 | 2.494 | 27.6129 | 66.1108 | 17.4375 |
| encode_decode_tiled | VAE encode+decode tiling may help both VAE phases | 0 | 299 | 121.1064 | 2.4689 | 28.8906 | 66.0785 | 17.4375 |
