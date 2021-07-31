import os
import cv2

current_dir = os.path.realpath(os.path.dirname(__file__))
model = cv2.dnn.readNet(
    os.path.join(current_dir, 'face-detection-0200.bin'),
    os.path.join(current_dir, 'face-detection-0200.xml'))


def face_detect(img, threshold=0.95, min_size=10):
    """
    input: array[height, width, channel]
    output:
        [
            [x_min, y_min, x_max, y_max]
        ]
    """
    blob = cv2.dnn.blobFromImage(img, 1, (256, 256))
    model.setInput(blob)
    outs = model.forward()
    height, width = img.shape[:2]
    rets = []
    for _, _, conf, x_min, y_min, x_max, y_max in outs[0][0].tolist():
        if conf > threshold:
            ret = [x_min * width, y_min * height, x_max * width, y_max * height]
            ret = [int(x) for x in ret]
            ret = {
                'conf': conf,
                'x_min': max(0, min(width, ret[0])),
                'y_min': max(0, min(height, ret[1])),
                'x_max': max(0, min(width, ret[2])),
                'y_max': max(0, min(height, ret[3]))
            }
            if ret['x_max'] - ret['x_min'] >= min_size and ret['y_max'] - ret['y_min'] >= min_size:
                rets.append(ret)
    return rets
