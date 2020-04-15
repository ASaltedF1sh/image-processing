import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img_npc = cv2.imread('C:/Users/swz/Desktop/img/606_image.png').astype(np.float)

# Display histogram
plt.hist(img_npc.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.savefig("out.png")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
