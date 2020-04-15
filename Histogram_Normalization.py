import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_normalization(img, a = 0, b = 255):

	H, W, C = img.shape

	p_max = np.max(img)
	p_min = np.min(img)

	out = np.zeros_like(img)

	for y in range(H):
		for x in range(W):
			for c in range(C):
				out[x,y,c] = ((img[x,y,c] - p_min)/(p_max - p_min) * (b - a) + a).astype(np.uint8)
	return out