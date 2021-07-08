import numpy as np
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
print(a, '\n')
h, w = a.shape
h_list = [i for i in range(h)]
w_list = [i for i in range(w)]

a_ = [[a[i, j] + 5 if a[i, j] >10 else a[i, j] for j in h_list] for i in w_list]
print(np.array(a_))

# b = [[n + m for n in range(4)] for m in range(0, 16, 4)]
# b = np.array(b)
# print(b)

# t = 5
# # print(a[(([]), ([]))])
# np.where(a == b)
# a[np.where(a == b)] = 0
# print(a)
# # print(v)