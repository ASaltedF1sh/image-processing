import numpy as np
import cv2


def hough(edge):

	H, W= edge.shape
	table_H = (np.sqrt(H ** 2 + W ** 2)).astype(np.int)
	table_W = 180
	table = np.zeros((2 * table_H, table_W), dtype = np.uint8)
	index = np.where(edge_output == 255)

	for y, x in zip(index[0], index[1]):
		for i in range(180):
			t = np.pi / 180. * i
			r = int(x * np.cos(t) + y * np.sin(t))
			table[r + table_H, i] += 1

	return table.astype(np.uint8)

def NMS(curve, K = 20):

	H, W = curve.shape
	new_curve = curve.copy()

	for y in range(H):
		for x in range(W):
			if x-1>=0 and x+1 <= W-1 and y-1>=0 and y+1 <= H-1:
				if curve[y,x] < max(curve[y,x-1], curve[y-1,x-1], curve[y+1,x-1], curve[y-1,x], \
									curve[y+1,x], curve[y-1,x+1],curve[y,x+1],curve[y+1,x+1]):
					new_curve[y,x] =0

	topK_ind = new_curve.ravel().argsort()[::-1][:K]
	ind_x, ind_y = np.unravel_index(topK_ind, curve.shape)
	new_curve = np.zeros_like(curve, dtype = np.uint8)
	new_curve[ind_x, ind_y] = 255

	return 	new_curve

def draw_line(vote_result, img):

	H, W, C= img.shape
	index = np.where(vote_result == 255)
	bias = (np.sqrt(H ** 2 + W ** 2)).astype(np.int)
	out = img.copy()

	for y, x in zip(index[0], index[1]):
		r = y - bias
		theta = np.pi / 180. * x
		#不分情况讨论的话，可能会丢失水平或者垂直方向的目标，但是会造成大量重复计算
		#如果对性能有要求，可以单独对水平和垂直方向做计算
		for i in range(W):
			if x!=0:
				h0 = int(-i / np.tan(theta) + r / np.sin(theta))
				if h0 <= H-1 and h0 >= 0:
					out[h0, i] = [0, 0, 255]

		for j in range(H):
			if x!= 90:
				w0 = int(-j * np.tan(theta) + r / np.cos(theta))
				if w0 <= W-1 and w0 >= 0:
					out[j, w0] = [0, 0, 255]
	return out 
