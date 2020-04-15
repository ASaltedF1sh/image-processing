import numpy as np
import matplotlib.pyplot as plt
import copy

def hist_equalization(img):

	H, W, C = img.shape

	out = copy.deepcopy(img)
	total_prob = 0.
	total_pixels = H * W * C

	for i in range(0, 255):
		total_prob += len(img[img==i].tolist()) / total_pixels
		new_value = 255 * total_prob
		out[img==i] = new_value

	return out.astype(np.uint8)
