# ##############################################################################
# Q3 : Binalization using nomal threshold
# ##############################################################################

# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------

import cv2
import numpy as np

# make function
def convert_gray(img):
    img = img.astype(np.float64)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    out = 0.2126*r + 0.7152*g + 0.0722*b
    return out.astype(np.uint8)

def binalization(img):
    th = 128
    img[img < th] = 0
    img[img >= th] = 255
    return img

# read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# convert to gray scale
img = convert_gray(img)
# convert to binalization
img = binalization(img)


cv2.imshow('imori', img)
cv2.waitKey(0)
cv2.destroyAllWindows

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------
# answer code
# ------------------------------------------------------------------------------
# import cv2
# import numpy as np

# # Gray scale
# def BGR2GRAY(img):
# 	b = img[:, :, 0].copy()
# 	g = img[:, :, 1].copy()
# 	r = img[:, :, 2].copy()

# 	# Gray scale
# 	out = 0.2126*r + 0.7152*g + 0.0722*b
# 	out = out.astype(np.uint8)

# 	return out

# # binalization
# def binalization(img, th=128):
# 	img[img < th] = 0  # ※※※
# 	img[img >= th] = 255 # ※※※
# 	return img
# # ※※※
# # この構文はndarrayの特定の要素を抽出可能な構文である．
# # ndarray[condition] で条件を満たすものだけを抽出し1次元に平坦化された配列ができる．
# # ndarray > th（閾値）とすると閾値をtrueとfalseで出来た配列(bool型配列)ができる
# # ndarray[condition] = value とすると，上に書いたような配列でtrueにだけ代入することができる．
# # （今回の場合condition が ndarray > th であった．）


# # Read image
# img = cv2.imread("imori.jpg").astype(np.float64)

# # Convert to Grayscale
# out = BGR2GRAY(img)

# # Convert to Binalization
# out = binalization(out)

# # Save result
# # cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ******************************************************************************