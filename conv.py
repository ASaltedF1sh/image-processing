import numpy as np
import cv2
import math
#padding_type : zero(default), replicate (复制最近的边界) , reverse

def max_pooling(crop):
	return np.max(crop).astype(np.uint8)

def median_pooling(crop):
	return np.median(crop).astype(np.uint8)

def average_pooling(crop):
	return np.average(crop).astype(np.uint8)

def gaussian_filter(crop, kernel_size=7, sigma_X=None, sigma_Y=None):

	assert isinstance(kernel_size, int) and (kernel_size % 2) != 0

	if sigma_X is None:
		sigma_X = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
	if sigma_Y is None:
		sigma_Y = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

	kernel_X = np.zeros(kernel_size, dtype=np.float32)
	kernel_Y = np.zeros(kernel_size, dtype=np.float32)
	for i in range(kernel_size - kernel_size//2):
		kernel_X[i] = 1/(np.sqrt(2*np.pi)*sigma_X)*np.exp(-((kernel_size//2-i)/sigma_X)**2/2)
		kernel_Y[i] = 1/(np.sqrt(2*np.pi)*sigma_Y)*np.exp(-((kernel_size//2-i)/sigma_Y)**2/2)
	for i in range(kernel_size//2):
		kernel_X[kernel_size - i -1] = kernel_X[i]
		kernel_Y[kernel_size - i -1] = kernel_Y[i]
	kernel_X /= np.sum(kernel_X)
	kernel_Y /= np.sum(kernel_Y)
	kernel_X = kernel_X.reshape(kernel_size,1)
	kernel_Y = kernel_Y.reshape(1,kernel_size)
	kernel = kernel_X * kernel_Y

	return np.sum(crop*kernel).astype(np.uint8)

def my_conv(img, func, kernel_size=3, padding='same', padding_type='zero', stride=2):

	assert kernel_size >=1 and isinstance(kernel_size, int) and stride >=1 and isinstance(stride, int)
	H, W, C = img.shape
	assert H >= kernel_size and W >= kernel_size and W > stride and H > stride

	if padding == 'same':

		if stride == 1:
			out = np.zeros_like(img)

			if padding_type == 'zero':
				padded_img = np.zeros((H+kernel_size-1, W+kernel_size-1, C), dtype=np.uint8)
				# for i, y in enumerate(range(kernel_size-1-(kernel_size-1)//2 , H+kernel_size-1-(kernel_size-1)//2)):
				# 	for j ,x in enumerate(range(kernel_size-1-(kernel_size-1)//2 , W+kernel_size-1-(kernel_size-1)//2)):
				# 		for c in range(C):
				# 			padded_img[y, x, c] = img[i,j,c]
				padded_img[kernel_size-1-(kernel_size-1)//2:H+kernel_size-1-(kernel_size-1)//2, kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2,:] = img

			elif padding_type == 'replicate':
				padded_img = np.zeros((H+kernel_size-1,W+kernel_size-1,C), dtype=np.uint8)
				padded_img[kernel_size-1-(kernel_size-1)//2:H+kernel_size-1-(kernel_size-1)//2, kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2,:] = img
				for i in range(kernel_size-1-(kernel_size-1)//2):
					padded_img[i, kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2, :] = img[0,:,:]
				for i in range(H+kernel_size-1-(kernel_size-1)//2, H+kernel_size-1):
					padded_img[i, kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2, :] = img[-1,:,:]

				for i in range(kernel_size-1-(kernel_size-1)//2):
					padded_img[:, i, :] = padded_img[:, kernel_size-1-(kernel_size-1)//2,:]
				for i in range(W+kernel_size-1-(kernel_size-1)//2, W+kernel_size-1):
					padded_img[:, i, :] = padded_img[:, W+kernel_size-1-(kernel_size-1)//2-1,:]

			for y in range(H):
				for x in range(W):
					for c in range(C):
						out[x,y,c] = func(padded_img[x:x+kernel_size, y:y+kernel_size, c])
		else:

			out = np.zeros((math.ceil(H/stride), math.ceil(W/stride), C), dtype=np.uint8)
			#由输出的目标尺寸倒推padding的尺寸
			residual_H = (math.ceil(H/stride) -1) * stride + kernel_size - H
			residual_W = (math.ceil(W/stride) -1) * stride + kernel_size - W
			if residual_W <0:
				residual_W = 0
			if residual_H <0:
				residual_H = 0
			padded_img = np.zeros((H+residual_H, W+residual_W, C), dtype=np.uint8)
			padded_img[residual_H//2:H+residual_H-(residual_H-residual_H//2), residual_W//2:W+residual_W-(residual_W-residual_W//2), :] = img

			if padding_type == 'zero':
				pass

			elif padding_type == 'replicate':
				for i in range(residual_H//2):
					padded_img[i, residual_W//2:W+residual_W-(residual_W-residual_W//2), :] = img[0,:,:]
				for i in range(H+residual_H-(residual_H-residual_H//2), H+residual_H):
					padded_img[i, residual_W//2:W+residual_W-(residual_W-residual_W//2), :] = img[-1,:,:]

				for i in range(residual_W//2):
					padded_img[:, i, :] = padded_img[:, residual_W//2,:]
				for i in range(W+residual_W-(residual_W-residual_W//2), W+residual_W):
					padded_img[:, i, :] = padded_img[:, W+residual_W-(residual_W-residual_W//2)-1,:]

			for i,y in enumerate(range(math.ceil(H/stride))):
				for j,x in enumerate(range(math.ceil(W/stride))):
					for c in range(C):
						out[x,y,c] = func(padded_img[j*stride:j*stride+kernel_size, i*stride:i*stride+kernel_size, c])


	if padding == 'valid':
		pass

	return out
