import numpy as np
import cv2

img_original = cv2.imread("Q1_Q10/imori.jpg")

img = img_original.copy()

img = img[:, :, 1]
img[img < 128] = 0
img[img >= 128] = 255

cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyWindow

# h, w, c = img.shape

# img[:h//2, :w//2] = img[:h//2, :w//2, ::-1]

# cv2.imshow('imori', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows
