import os
import cv2
import numpy as np


current_dir = os.path.realpath(os.path.dirname(__file__))
model = cv2.dnn_DetectionModel(
    os.path.join(current_dir, 'person-detection-retail-0002.bin'),
    os.path.join(current_dir, 'person-detection-retail-0002.xml'))


def person_detect(img, threshold=0.5, min_size=10):
    """
    input: array[height, width, channel]
    output:
        image
    """
    
    outs = model.detect(img)
    _, conf, pos = outs
    height, width = img.shape[:2]
    rets = []
    if len(conf) > 0:
        for c, (x_min, y_min, w, h) in zip(conf.flatten(), pos):
            if c < threshold:
                continue
            x_max = x_min + w
            y_max = y_min + h
            x_min = int(x_min * img.shape[1] / 992)
            x_max = int(x_max * img.shape[1] / 992)
            y_min = int(y_min * img.shape[0] / 544)
            y_max = int(y_max * img.shape[0] / 544)
            ret = {
                'conf': c,
                'x_min': max(0, min(width, x_min)),
                'y_min': max(0, min(height, y_min)),
                'x_max': max(0, min(width, x_max)),
                'y_max': max(0, min(height, y_max))
            }
            if ret['x_max'] - ret['x_min'] >= min_size and ret['y_max'] - ret['y_min'] >= min_size:
                rets.append(ret)

    return rets
