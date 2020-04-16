import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def dilate(img, times):

	def my_conv(img, filter, kernel_size=3, padding='same', padding_type='zero', stride=1):

		assert kernel_size >=1 and isinstance(kernel_size, int) and stride >=1 and isinstance(stride, int) and (kernel_size % 2) != 0

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True
		img = img.copy().astype(np.float)
		assert H >= kernel_size and W >= kernel_size and W > stride and H > stride

		if padding == 'same':

			if stride == 1:
				out = np.zeros_like(img, dtype = np.float)

				if padding_type == 'zero':
					padded_img = np.zeros((H+kernel_size-1, W+kernel_size-1, C), dtype=np.float)
					padded_img[kernel_size-1-(kernel_size-1)//2:H+kernel_size-1-(kernel_size-1)//2, \
						   kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2,:] = img

				elif padding_type == 'replicate':
					pass

				for y in range(H):
					for x in range(W):
						for c in range(C):
							out[y,x,c] = filter(padded_img[y:y+kernel_size, x:x+kernel_size, c])
			else:
				pass
		else:
			pass

		return out.astype(np.uint8)

	def Otsu(img):

		H, W, C = img.shape

		B = img[:,:,0]
		G = img[:,:,1]
		R = img[:,:,2]

		out =  0.299 * R + 0.587 * G + 0.114 * B
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

		return out

	def dilate_filter(cropped):

		kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
		result = np.sum(kernel * cropped)
		if result >= 255:
			return 255
		else:
			return 0

	binary_img = Otsu(img)
	dilated_img = binary_img

	for i in range(times):
		dilated_img = my_conv(dilated_img, dilate_filter)

	return np.squeeze(dilated_img)
