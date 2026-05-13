# Integrated W4A4 TensorRT MLP Inference

This run routes the first DiT video MLP branch through an integrated ModelOpt NVFP4 W4A4 TensorRT `DeviceModel` inside the actual 81-frame inference path.

| case | rc | frames | seconds | fps |
| --- | ---: | ---: | ---: | ---: |
| trt_w4a4_integrated_mlp | 0 | 81 | 85.86 | 0.9434 |

Compile/runtime details:

- subgraph: `dit.blocks[0].mlp.vid`
- boundary: `x[74655, 3072] -> y[74655, 3072]`
- parameters: `75,512,832`
- input dtype seen in the live DiT block: `torch.float32`
- TensorRT compile dtype: `torch.float16`
- QAT: one SGD teacher-loss step, loss `0.005459692794829607`
- ModelOpt W4A4 deploy/TRT profile: `30.5263ms`, `32.7569` inferences/s
- integrated compile time: `45.2335s`
- max absolute error against live teacher activation: `0.75`

Artifacts:

- `trt_w4a4_integrated_mlp.json`
- `trt_w4a4_integrated_mlp_compile.json`
- `trt_w4a4_integrated_mlp.log`
