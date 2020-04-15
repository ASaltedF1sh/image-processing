import numpy as np
import cv2


def Otsu(img):
	H, W, C = img.shape
	assert C>=1,"img input should be a rgb image"

	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]

	out = 0.2126 * R + 0.7152 * G + 0.0722 * B
	out = out.astype(np.uint8)

	best_threshold = np.min(out)
	intra_viarance = 0
	total_pixels = H*W
	for gray_scale in range(np.min(out), np.max(out)+1):
		w0 = len(out[out>gray_scale].tolist()) / total_pixels
		w1 = 1 - w0
		M0 = np.average(out[out>=gray_scale].tolist())
		M1 = np.average(out[out<gray_scale].tolist())
		if w0 * w1 * ((M0 - M1) ** 2) > intra_viarance :
			intra_viarance = w0 * w1 * ((M0 - M1) ** 2)
			best_threshold = gray_scale

	out[out<best_threshold] = 0
	out[out>=best_threshold] = 255
	return out, best_threshold