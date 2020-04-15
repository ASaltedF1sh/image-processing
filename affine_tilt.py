import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Affine
def affine(img, dx=30, dy=30):
    # get shape
    assert dx/90 %2 !=1 and dy/90 %2 !=1,'valid input!'
    H, W, C = img.shape

    # Affine hyper parameters
    a = 1.
    b = math.tan(dx*math.pi/180.0)
    c = math.tan(dy*math.pi/180.0)
    d = 1.
    tx = 0.
    ty = 0.

    # prepare temporary
    _img = np.zeros((H+2, W+2, C), dtype=np.float32)
    _img[1:H+1, 1:W+1] = img

    #要考虑角度为正为负不同情况的bounding box
    W_left = min(np.ceil(b*H).astype(np.int), 0, W+np.ceil(b*H).astype(np.int), W)
    W_right = max(np.ceil(b*H).astype(np.int), 0, W+np.ceil(b*H).astype(np.int), W)
    H_up = max(np.ceil(c*W).astype(np.int), 0, H+np.ceil(c*W).astype(np.int), H)
    H_down = min(np.ceil(c*W).astype(np.int), 0, H+np.ceil(c*W).astype(np.int), H)
    #计算得到仿射变换后图像bounding box的长和宽
    W_new = W_right - W_left + 1
    H_new = H_up - H_down + 1

    out = np.zeros((H_new, W_new, C), dtype=np.float32)

    # preprare assigned index
    #（x_new和y_new 分别是对应仿射变换后图形的bounding box中各点的坐标
    x_new = np.tile(np.arange(W_left, W_right+1), (H_new, 1))
    y_new = np.arange(H_down,H_up+1).repeat(W_new).reshape(H_new, -1)

    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    out = _img[y, x]
    out = out.astype(np.uint8)

    return out

