# ##############################################################################
# Q7 : Average pooling
# This operation plays a great role on CNN.
# ##############################################################################


# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------
import cv2
import numpy as np
from numpy.lib.function_base import average

def avg_pooling(img, x):
    img = img.astype(np.float64)
    h, w, d = img.shape
    out = img

    # /////////////////////////////////////////////////////////////////////////
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # poolingは平均化したものをその領域全てに代入するもの．
    # この手法でも一見平均化出来てそうだが，これによりできるimgはサイズが256×256で
    # はなく，32×32（x=8の場合）となってしまう．
    # つまり画像サイズそのものが小さくなってしまっているため引き延ばした際にモザイ
    # クっぽくなっているだけ．
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # B = img[:, :, 0]
    # G = img[:, :, 1]
    # R = img[:, :, 2]
    # B_ = np.array([[np.average(B[i:i+x,j:j+x]) for j in range(0, w, x)] for i in range(0, h, x)])
    # G_ = np.array([[np.average(G[i:i+x,j:j+x]) for j in range(0, w, x)] for i in range(0, h, x)])
    # R_ = np.array([[np.average(R[i:i+x,j:j+x]) for j in range(0, w, x)] for i in range(0, h, x)])
    # /////////////////////////////////////////////////////////////////////////

    for i in range(1, w, x):
        for j in range(1, h, x):
            for k in range(1, d):
                out[i:i+x, j:j+x, k] = np.mean([out[i:i+x, j:j+x, k]])

    return out.astype(np.uint8)

# Read image and pooling range
X = int(input("Input range using in pooling process (What is x for x*x ?) : "))
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# Call function for image
img = avg_pooling(img, X)
print(img.shape)
img = cv2.resize(img, (256, 256))

# Show result
cv2.imshow('imori', img)
cv2.waitKey(0)
cv2.destroyAllWindows

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------
# answer code
# ------------------------------------------------------------------------------
# import cv2
# import numpy as np


# # average pooling
# def average_pooling(img, G=8):
#     out = img.copy()

#     H, W, C = img.shape
#     print(img.shape)
#     Nh = int(H / G)
#     Nw = int(W / G)

#     for y in range(Nh):
#         for x in range(Nw):
#             for c in range(C):
#                 out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)

#     return out


# # Read image
# img = cv2.imread("imori.jpg")

# # Average Pooling
# out = average_pooling(img)

# # Save result
# # cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ******************************************************************************