import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#padding_type : ZERO(default), REPLICATE (复制最近的边界)
def my_conv(img, kernel, kernel_size=3, padding='same', padding_type='replicate', stride=1):

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
				padded_img[kernel_size-1-(kernel_size-1)//2:H+kernel_size-1-(kernel_size-1)//2, kernel_size-1-(kernel_size-1)//2:W+kernel_size-1-(kernel_size-1)//2,:] = img

			elif padding_type == 'replicate':
				padded_img = np.zeros((H+kernel_size-1,W+kernel_size-1,C), dtype=np.float)
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
						out[y,x,c] = func(padded_img[y:y+kernel_size, x:x+kernel_size, c])

	return out

def BGR2Gray(img):
	H, W, C = img.shape
	assert C>=1,"img input should be a rgb image"

	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]

	out = 0.299 * R + 0.587 * G + 0.114 * B
	out = out.astype(np.uint8)

	return out

def gaussian_filter(img, kernel_size=3, sigma_X=None,sigma_Y=None):

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
	# return my_conv(img, kernel, kernel_size)
	return np.clip(my_conv(img, kernel, kernel_size), 0, 255).astype(np.uint8)

def sobel_filter(img, direction = 'X'):

	assert direction == 'X' or direction == 'Y'
	if direction == 'X':
		kernel = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
	elif direction == 'Y':
		kernel = np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])

	return my_conv(img, kernel, 3)

def get_gradient_amplitude_and_angel(x,y):

	amp = np.sqrt(x **2 + y **2)
	angle = np.arctan2(x ,y) * 180. / np.pi

	angle[(((67.5<angle) & (angle<=112.5)) | ((-112.5<angle) & (angle<=-67.5)))] = 1
	angle[(((22.5<angle) & (angle<=67.5))  | ((-67.5<angle) & (angle<=-22.5)))] = 2
	angle[(((112.5<angle) & (angle<=157.5)) | ((-157.5<angle) & (angle<=-112.5)))] = 3
	angle[(((0<angle) & (angle<=22.5)) | ((157.5<angle) & (angle<=180)) | ((-22.5<angle) & (angle<=0)) | ((-180<=angle) & (angle<=-157.5)))] = 4

	return  amp, angle

def NMS(amp, ang):

	H, W ,C= amp.shape
	new_amp = amp.copy()
	for x in range(H):
		for y in range(W):
			if ang[x,y] == 1:
				if y-1>=0 and y+1 <= H-1:
					if amp[x,y] != max(amp[x,y-1], amp[x,y], amp[x,y+1]):
						new_amp[x,y] =0
			elif ang[x,y] == 2:
				if x-1>=0 and x+1 <= W-1 and y-1>=0 and y+1 <= H-1:
					if amp[x,y] != max(amp[x-1,y-1], amp[x,y], amp[x+1,y+1]):
						new_amp[x,y] =0
			elif ang[x,y] == 3:
				if x-1>=0 and x+1 <= W-1 and y-1>=0 and y+1 <= H-1:
					if amp[x,y] != max(amp[x-1,y+1], amp[x,y], amp[x+1,y-1]):
						new_amp[x,y] =0				
			elif ang[x,y] == 4:
				if x-1>=0 and x+1 <= W-1:
					if amp[x,y] != max(amp[x-1,y], amp[x,y], amp[x+1,y]):
						new_amp[x,y] =0
	return new_amp

def Hysteresis_thresholding(amp, HT=150, LT=100):
	# plt.hist(amp.ravel(), bins=20, rwidth=0.8)
	# plt.show()
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	H, W ,C= amp.shape
	new_amp = amp.copy().astype(np.uint8)
	for x in range(H):
		for y in range(W):
			if amp[x,y]>= HT:
				new_amp[x,y] = 255
			elif amp[x,y]<= LT:
				new_amp[x,y] = 0
			elif  LT< amp[x,y]< HT:
				if x-1>=0 and x+1 <= W-1 and y-1>=0 and y+1 <= H-1:
					if HT <= max(amp[x-1,y+1], amp[x+1,y-1],amp[x-1,y-1], amp[x+1,y+1], amp[x-1,y], amp[x+1,y], amp[x,y-1], amp[x,y+1]):
						new_amp[x,y] = 255
					else:
						new_amp[x,y] = 0
	return new_amp


def nothing(x):
	pass
img = cv2.imread("C:/Users/swz/Desktop/lena.jpg").astype(np.float32)

denoised = gaussian_filter(img, 3)
gray = BGR2Gray(denoised)

sobel_x = sobel_filter(gray, 'X')
sobel_y = sobel_filter(gray, 'Y')

amp, angle = get_gradient_amplitude_and_angel(sobel_x, sobel_y)
new_amp = NMS(amp, angle)
edge=np.squeeze(Hysteresis_thresholding(new_amp))

cv2.namedWindow('res')
cv2.createTrackbar('min','res',0,255,nothing)
cv2.createTrackbar('max','res',0,255,nothing)
while(1):
	cv2.imshow('edge',edge)

	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	maxVal=cv2.getTrackbarPos('max','res')
	minVal=cv2.getTrackbarPos('min','res')
	edge=np.squeeze(Hysteresis_thresholding(new_amp, maxVal, minVal))
cv2.destroyAllWindows()

# src = cv2.imread("C:/Users/swz/Desktop/lena.jpg")
# blurred = cv2.GaussianBlur(src, (3, 3), 0)
# grayed = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
# edge_output = cv2.Canny(grayed, 50, 150)
# imgs = np.hstack([gray.astype(np.uint8), edge,edge_output])

# cv2.imshow("result", imgs)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
