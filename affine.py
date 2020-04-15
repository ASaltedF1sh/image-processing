import cv2
import numpy as np
import matplotlib.pyplot as plt


# Affine
def affine(img, a, b, c, d, tx, ty):
	H, W, C = img.shape
	_img = img.copy()
	img = np.zeros((H+2, W+2, C), dtype=np.float32)
	img[1:H+1, 1:W+1] = _img
	H_new = np.round(H * d).astype(np.int)
	W_new = np.round(W * a).astype(np.int)
	out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image

	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1
	#限定对应原图的坐标值范围最小是0，最大是边长加+1（因为对边长补了1）
	#这样很显然超出范围的,即在原图中没有对应点的将变成黑色的区块
	x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
	y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixcel to new image
	out[y_new, x_new] = img[y, x]

	out = out[:H_new, :W_new]
	out = out.astype(np.uint8)

	return out

