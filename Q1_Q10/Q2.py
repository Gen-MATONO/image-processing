##############################################
# --------------------------------------------
# THE ANSWER PATURN WHICH USE SKIMAGE AND PLT
# --------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io

# img_original = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/assets/imori_256x256.png')

# img = img_original.copy().astype(np.float64)

# r = img[..., 0]
# g = img[..., 1]
# b = img[..., 2]

# gray = 0.2126*r + 0.7152*g + 0.0722*b

# gray = gray.astype(np.uint8)

# plt.figure()
# ax = plt.subplot()
# ax.imshow(gray, cmap = 'gray')
# plt.show()

##############################################
# --------------------------------------------
# THE ANSWER PATURN WHICH USE ONLY OPENCV
# --------------------------------------------
import cv2
import numpy as np

img_original = cv2.imread('imori.jpg')
img = img_original.copy().astype(np.float64)

b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

gray = 0.2126*r + 0.7152*g + 0.0722*b

gray = gray.astype(np.uint8)

cv2.imshow('imori', gray)
cv2.waitKey(0)
cv2.destroyAllWindows

##############################################

##############################################
# ANSWER
##############################################

# import cv2
# import numpy as np

# # Gray scale
# def BGR2GRAY(img):
# 	b = img[:, :, 0].copy()
# 	g = img[:, :, 1].copy()
# 	r = img[:, :, 2].copy()

# 	# Gray scale
# 	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
# 	out = out.astype(np.uint8)

# 	return out


# # Read image
# img = cv2.imread("imori.jpg").astype(np.float64)

# # Grayscale
# out = BGR2GRAY(img)

# # Save result
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()