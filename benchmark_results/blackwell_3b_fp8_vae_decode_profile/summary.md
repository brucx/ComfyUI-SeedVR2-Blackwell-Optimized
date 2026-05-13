# 3B FP8 VAE Decode Profile Summary

| Case | Frames | Seconds | FPS | VAE encode | VAE decode phase | Inner vae_decode | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| profile_default | 81 | 57.0813 | 1.4190 | 7.4633 | 41.1475 | 40.6648 | 20.22 GB |
| profile_convlimit_off | 81 | 37.5084 | 2.1595 | 6.6392 | 22.2895 | 21.7939 | 44.44 GB |
| convlimit_off validation | 81 | 28.6162 | 2.8306 | 6.5851 | 13.5041 | 11.6760 | 40.97 GB |
| preset_convlimit_default | 81 | 28.2877 | 2.8634 | 6.5334 | 13.3488 | 11.5937 | 40.97 GB |
| full_convlimit_off | 299 | 103.0863 | 2.9005 | 23.8534 | 53.5754 | 11.9642/logged batch | 41.84 GB |

Profiler traces are intentionally not stored in git because each Chrome trace is about 140 MB. The committed profiler artifacts are the CUDA tables and top-op JSON files.
