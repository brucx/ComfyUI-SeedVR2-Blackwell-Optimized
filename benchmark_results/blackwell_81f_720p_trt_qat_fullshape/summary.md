# Full-Shape TRT W4A4 QAT Probe

This run uses the real fixed token count for the requested 81-frame 720p benchmark:

- encoded latent shape: `[21, 90, 158, 16]`
- DiT patch size: `[1, 2, 2]`
- video MLP token length: `21 * 45 * 79 = 74655`
- subgraph: `dit.blocks[0].mlp.vid`
- boundary: `x[74655, 3072] -> y[74655, 3072]`
- parameters: `75,512,832`

| case | status | latency |
| --- | --- | ---: |
| FP16 eager | ok | 48.8109ms |
| FP16 Torch-TensorRT | ok | 44.5086ms |
| FP16 ModelOpt deploy/TRT | ok | 45.8629ms |
| W4A4 eager after 1-step QAT | ok | 48.6992ms |
| W4A4 ModelOpt deploy/TRT profile | ok | 31.9918ms |
| W4A4 ModelOpt `DeviceModel` forward | ok | 30.9279ms |

The one-step W4A4 QAT teacher loss was `2.1344993114471436` using SGD. The finite W4A4 TensorRT output had max absolute error `71.0` against the FP16 teacher for the random full-shape probe batch.

The JSON artifact is `trt_w4a4_qat_probe.json`.
