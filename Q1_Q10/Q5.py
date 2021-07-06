# ##############################################################################
# Q5 : HSV conversion
# H(Hue) : classification of color using 1 to 360
# S(Saturation) : Vividness of color
# V(Value) : Brightness of color
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

    # Invert hue
    H = H + 180
    H = [] # I don't know how to handle the value over 360

    # There are no errors now.

# Read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------
# answer code
# ------------------------------------------------------------------------------

# ******************************************************************************