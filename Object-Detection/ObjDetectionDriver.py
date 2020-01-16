from PIL import Image, ImageGrab
import cv2
import time
import numpy as np
import ObjDetectionCore
np.seterr(all='ignore')

frameNum = 0

vid = cv2.VideoCapture('challenge.mp4')
suc=1
start = time.time()
while suc:
    frameNum += 1
    suc, image=vid.read()
    fps = round(frameNum / (time.time()-start),1)
    image = ObjDetectionCore.predict(image)
    cv2.putText(image, str(fps), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 2)
    cv2.imshow('objd',image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # predict(image)
cv2.destroyAllWindows()
