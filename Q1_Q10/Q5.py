# ##############################################################################
# Q5 : HSV conversion
# H(Hue) : classification of color using 1 to 360
# S(Saturation) : Vividness of color
# V(Value) : Brightness of color
# I can't write this program well.
# I will skip this question. If you(you are me in the future) feel like, you will
# try this question again.
# ##############################################################################

# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------
import cv2
import numpy as np

def hsv(img):
    img = img.astype(np.float64)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    Max = np.maximum(R  ,G)
    Max = np.maximum(Max,B)
    Min = np.minimum(R  ,G)
    Min = np.minimum(Min,B)

    # Convert RGB to HSV
    V = Max

    S = Max - Min

    H = Max
    H[np.where(Min == Max)] = 0
    H[np.where(Min == B  )] = 60*(G - R)/(Max-Min) + 60
    H[np.where(Min == R  )] = 60*(B - G)/(Max-Min) + 180
    H[np.where(Min == G  )] = 60*(R - B)/(Max-Min) + 300

    # make index list
    horizontal, vertical = H.shape
    v_list = [i for i in range(vertical)]
    h_list = [j for j in range(horizontal)]

    # Invert hue
    H += 180
    # for m in v_list:
    #     for n in h_list:
    #         if H[m, n] > 360:
    #             H[m, n] -= 360

    # Retrun HSV to RGB
    C = S
    H_ = H/60
    X = C (1 - np.abs(H_ % 2 - 1))

    out = [V - C]
    out = [[out[i, j, 0] + C[i, j] if 0 <= H_(i, j) < 1 or 5 <= H_(i, j) < 6 else out(i, j) for j in v_list] for i in h_list]
    out = [[out[i, j, 1] + X[i, j] if 0 <= H_(i, j) < 1 or 3 <= H_(i, j) < 4 else out(i, j) for j in v_list] for i in h_list]
    out = [[out[i, j, 0] + X[i, j] if 1 <= H_(i, j) < 2 or 4 <= H_(i, j) < 5 else out(i, j) for j in v_list] for i in h_list]
    out = [[out[i, j, 1] + C[i, j] if 1 <= H_(i, j) < 2 or 2 <= H_(i, j) < 3 else out(i, j) for j in v_list] for i in h_list]
    out = [[out[i, j, 2] + X[i, j] if 2 <= H_(i, j) < 3 or 5 <= H_(i, j) < 6 else out(i, j) for j in v_list] for i in h_list]
    out = [[out[i, j, 2] + C[i, j] if 3 <= H_(i, j) < 4 or 4 <= H_(i, j) < 4 else out(i, j) for j in v_list] for i in h_list]
    out = np.array(out)

    return out.astype(np.uint8)

# Read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

img = hsv(img)

cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------
# answer code
# ------------------------------------------------------------------------------
# import cv2
# import numpy as np


# # BGR -> HSV
# def BGR2HSV(_img):
# 	img = _img.copy() / 255.

# 	hsv = np.zeros_like(img, dtype=np.float32)

# 	# get max and min
# 	max_v = np.max(img, axis=2).copy()
# 	min_v = np.min(img, axis=2).copy()
# 	min_arg = np.argmin(img, axis=2)

# 	# H
# 	hsv[..., 0][np.where(max_v == min_v)]= 0
# 	## if min == B
# 	ind = np.where(min_arg == 0)
# 	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
# 	## if min == R
# 	ind = np.where(min_arg == 2)
# 	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
# 	## if min == G
# 	ind = np.where(min_arg == 1)
# 	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

# 	# S
# 	hsv[..., 1] = max_v.copy() - min_v.copy()

# 	# V
# 	hsv[..., 2] = max_v.copy()

# 	return hsv


# def HSV2BGR(_img, hsv):
# 	img = _img.copy() / 255.

# 	# get max and min
# 	max_v = np.max(img, axis=2).copy()
# 	min_v = np.min(img, axis=2).copy()

# 	out = np.zeros_like(img)

# 	H = hsv[..., 0]
# 	S = hsv[..., 1]
# 	V = hsv[..., 2]

# 	C = S
# 	H_ = H / 60.
# 	X = C * (1 - np.abs( H_ % 2 - 1))
# 	Z = np.zeros_like(H)

# 	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

# 	for i in range(6):
# 		ind = np.where((i <= H_) & (H_ < (i+1)))
# 		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
# 		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
# 		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

# 	out[np.where(max_v == min_v)] = 0
# 	out = np.clip(out, 0, 1)
# 	out = (out * 255).astype(np.uint8)

# 	return out

# # Read image
# img = cv2.imread("imori.jpg").astype(np.float64)

# # RGB > HSV
# hsv = BGR2HSV(img)

# # Transpose Hue
# hsv[..., 0] = (hsv[..., 0] + 180) % 360

# # HSV > RGB
# out = HSV2BGR(img, hsv)

# # Save result
# # cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ******************************************************************************