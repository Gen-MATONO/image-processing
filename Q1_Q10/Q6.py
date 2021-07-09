# ##############################################################################
# Q6 : Color reduction processing
# In this question, I'd like to reduce color value 256**3 to 4**3.
# ##############################################################################


# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------
import cv2
import numpy as np

def red_color(img):
    img = img.astype(np.float64)

    # imgは256までだから4で割ると64．
    # つまり255までを64で割って切り捨てたものに64をかけると0, 64, 128, 192のどれか
    # 減色処理はそれぞれの色のレンジの中間値を取っているから32を足すと完了する．
    out = img//64*64 + 32

    return out.astype(np.uint8)

# Read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# Call function
img = red_color(img)

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


# # Decrease color
# def decrease_color(img):
# 	out = img.copy()

# 	out = out // 64 * 64 + 32

# 	return out


# # Read image
# img = cv2.imread("imori.jpg")

# # Dicrease color
# out = dicrease_color(img)

# # cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ******************************************************************************