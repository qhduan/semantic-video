import os
import cv2
import numpy as np


current_dir = os.path.realpath(os.path.dirname(__file__))
model = cv2.dnn.readNet(
    os.path.join(current_dir, 'facenet-20180408-102900.bin'),
    os.path.join(current_dir, 'facenet-20180408-102900.xml'))


def facenet(img):
    """
    input: array[height, width, channel]
    output:
        image
    """
    blob = cv2.dnn.blobFromImage(img, 1, (160, 160))
    model.setInput(blob)
    outs = model.forward(model.getUnconnectedOutLayersNames())
    return outs[0][0]
