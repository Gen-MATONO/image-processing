# ##############################################################################
# Q4 : Otsu's binalization
# Reading Gasyori100knock in github and the following site makes easy
# understanding for Otsu's binalization.
# https://kazenoha.hatenablog.com/entry/2019/04/14/225643
# The site title is "化学機械とかあれこれ，大津の二値化(判別分析法)の式変形とか"
# ##############################################################################

# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------
import cv2
import numpy as np

# make function related to convert image to gray scale
def convert_gray(img):
    img = img.astype(np.float64)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    out = 0.2126*r + 0.7152*g + 0.0722*b
    return out.astype(np.uint8)

# make function related to Otsu algorithm
def otsu(img):

    # classification and getting threshold
    sigma_max = 0
    t_max = 0

    for t in range(1, 256):
        pix0 = img[np.where(img < t)]
        pix1 = img[np.where(img >= t)]

        ave0 = np.average(pix0) if len(pix0) > 0 else 0.
        ave1 = np.average(pix1) if len(pix1) > 0 else 0.

        num0 = len(pix0)
        num1 = len(pix1)

        sigma = (num0*num1/(num0 + num1)**2)*((ave0 - ave1)**2)

        if sigma_max < sigma:
            sigma_max = sigma
            t_max = t

    print('threshold is {}'.format(t_max))

    # get binalization using threshold which was obtained in the previos section
    th = t_max
    img[img < th] = 0
    img[img >= th] = 255
    return img


# read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# convert to gray scale image
img = convert_gray(img)

# convert to binalization using Otsu's algorithm
img = otsu(img)

# show image
cv2.imshow('Otsu', img)
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
#     b = img[:, :, 0].copy()
#     g = img[:, :, 1].copy()
#     r = img[:, :, 2].copy()

#     # Gray scale
#     out = 0.2126 * r + 0.7152 * g + 0.0722 * b
#     out = out.astype(np.uint8)

#     return out

# # Otsu Binalization
# def otsu_binarization(img):
#     max_sigma = 0
#     max_t = 0
#     H, W = img.shape

#     # determine threshold
#     for _t in range(1, 256):
#         # v0 and v1 are the array of pixels in class1 and class2, respectively.
#         v0 = img[np.where(img < _t)] # np.where(cond)はcondを満たすときのindexをタプルで返す．
#                                      # img[((x),(y))]は二重タプルによる，配列の座標指定，(x, y)のimgを返す．
#         v1 = img[np.where(img >= _t)]

#         # m0 and m1 are the average of each classes, respectively.
#         m0 = np.mean(v0) if len(v0) > 0 else 0.
#         m1 = np.mean(v1) if len(v1) > 0 else 0.

#         # w0 and w1 are the number of of each classes, respectively.
#         # There are normalized by all pixels(H * W).
#         w0 = len(v0) / (H * W)
#         w1 = len(v1) / (H * W)

#         # Sigma is distribution between two classes.
#         sigma = w0 * w1 * ((m0 - m1) ** 2)

#         # Otsu's algorithm determines the threshold when sigma is max value.
#         if sigma > max_sigma:
#             max_sigma = sigma
#             max_t = _t

#     # Binarization
#     print("threshold >>", max_t)
#     th = max_t
#     img[img < th] = 0
#     img[img >= th] = 255

#     return out


# # Read image
# img = cv2.imread("imori.jpg").astype(np.float64)

# # Grayscale
# out = BGR2GRAY(img)

# # Otsu's binarization
# out = otsu_binarization(out)

# # Save result
# # cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ******************************************************************************