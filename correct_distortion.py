import numpy as np
import cv2

def correct_distortion(img):
    width= img.shape[1]
    height = img.shape[0]
    distCoeff = np.zeros((4,1),np.float64)

    k1 = -2.0e-6
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0
    cam[1,2] = height/2.0
    cam[0,0] = 10.
    cam[1,1] = 10.

    return cv2.undistort(img,cam,distCoeff)