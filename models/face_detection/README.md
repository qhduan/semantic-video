
https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/face-detection-0200/FP16-INT8/

https://github.com/openvinotoolkit/open_model_zoo/blob/b1ff98b64a6222cf6b5f3838dc0271422250de95/models/intel/face-detection-0200/README.md

## Inputs

Image, name: `input`, shape: `1, 3, 256, 256` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (0 - face)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner


