import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def nearest_neighbor_inter(img, ax = 1, ay = 1):

	H, W, C = img.shape

	out = np.zeros((round(H*ax), round(W*ay), C), dtype = np.float32)
	for y in range(round(H*ay)):
		for x in range(round(W*ax)):
			for c in range(C):
				out[x,y,c] = img[int(x//ax), int(y//ay),  c]

	return out.astype(np.uint8)