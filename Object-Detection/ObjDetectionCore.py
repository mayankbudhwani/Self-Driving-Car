# -*- coding: utf-8 -*-

import os
os.chdir('YOLO-master/YOLO-Image-labelling')
from darkflow.net.build import TFNet
options = {
    'model': 'yolo.cfg',
    'load': 'yolov2.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

import cv2


def predict(image):
    image_height, image_width, _ = image.shape
    result = tfnet.return_predict(image)
    for x in range(len(result)):
        tl = (result[x]['topleft']['x'], result[x]['topleft']['y'])
        br = (result[x]['bottomright']['x'], result[x]['bottomright']['y'])
        label = result[x]['label']
        image = cv2.rectangle(image, tl, br, (0, 255, 0), 4)
        image = cv2.putText(image, label, tl, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 4)
    return image
