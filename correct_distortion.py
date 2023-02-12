import numpy as np
import cv2

src= cv2.imread("./data/images/frame.png")
width= src.shape[1]
height = src.shape[0]

distCoeff = np.zeros((4,1),np.float64)

k1 = -1.0e-5; 
k2 = 0.0;
p1 = 0.0;
p2 = 0.0;

distCoeff[0,0] = k1;
distCoeff[1,0] = k2;
distCoeff[2,0] = p1;
distCoeff[3,0] = p2;

cam = np.eye(3,dtype=np.float32)

cam[0,2] = width/2.0
cam[1,2] = height/2.0
cam[0,0] = 10.
cam[1,1] = 10.

dst = cv2.undistort(src,cam,distCoeff)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()