import numpy as np
import cv2


def BGR2RGB(img):
	H, W, C = img.shape
	assert C>=1,"img input should be a rgb image"

	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]

	out = np.zeros((H,W,C),dtype = np.uint8)
	out[:,:,0] = R
	out[:,:,1] = G
	out[:,:,2] = B

	return out