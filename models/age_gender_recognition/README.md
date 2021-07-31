
https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/age-gender-recognition-retail-0013/FP16-INT8/

https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/README.md


## Inputs

Image, name: `input`, shape: `1, 3, 62, 62` in `1, C, H, W` format, where:

- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

1. Name: `age_conv3`, shape: `1, 1, 1, 1` - Estimated age divided by 100.
2. Name: `prob`, shape: `1, 2, 1, 1` - Softmax output across 2 type classes [0 - female, 1 - male].

