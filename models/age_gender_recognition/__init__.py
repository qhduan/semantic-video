
import os
import cv2
import numpy as np


current_dir = os.path.realpath(os.path.dirname(__file__))
model = cv2.dnn.readNet(
    os.path.join(current_dir, 'age-gender-recognition-retail-0013.bin'),
    os.path.join(current_dir, 'age-gender-recognition-retail-0013.xml'))


def age_gender_recognize(img):
    """
    input: array[height, width, channel]
    output:
        age, gender
    """
    blob = cv2.dnn.blobFromImage(img, 1, (62, 62))
    model.setInput(blob)
    age, gender = model.forward(model.getUnconnectedOutLayersNames())
    gender_prob = np.reshape(gender, (2,))
    gender_label = gender_prob.argmax()
    return {
        'age': int(float(np.reshape(age, (1,))[0]) * 100),
        'gender_label': gender_label,
        'gender': 'female' if gender_label == 0 else 'male',
        'female_prob': gender_prob[0],
        'male_prob': gender_prob[1]
    }
    return outs
