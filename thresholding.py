import numpy as np
import cv2


def thresholding(img):
	H, W, C = img.shape
	assert C>=1,"img input should be a rgb image"

	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]

	out = 0.2126 * R + 0.7152 * G + 0.0722 * B
	out = out.astype(np.uint8)
	out[out<128] = 0
	out[out>=128] = 255

	return out