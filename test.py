import numpy as np

from numpy.lib.function_base import average
import cv2

# def convert_gray(img):
#     img = img.astype(np.float64)
#     b = img[:, :, 0]
#     g = img[:, :, 1]
#     r = img[:, :, 2]
#     out = 0.2126*r + 0.7152*g + 0.0722*b
#     return out.astype(np.uint8)

# img_original = cv2.imread('Q1_Q10/imori.jpg')
# img = img_original.copy()
# gray = convert_gray(img)
# print(gray.shape)

# a = [(i, j) np.arange(1, 20, 2)]
# print(a[])
# print(average(a[1:4]))

a = [[i + j for i in range(4)] for j in range(0, 16, 4)]
a = np.array(a)

print(a)
t = 0
print(a[(([]), ([]))])
v = a[np.where(a < t)]
print(v)