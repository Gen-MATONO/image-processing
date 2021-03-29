from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt


img = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/assets/imori_256x256.png')

r = img[..., 0]
g = img[..., 1]
b = img[..., 2]

plt.figure()
ax = plt.subplot(1, 3, 1)
ax.set_title('R')
ax.imshow(r, cmap = "gray")
ax = plt.subplot(1, 3, 2)
ax.set_title('G')
ax.imshow(g, cmap = "gray")
ax = plt.subplot(1, 3, 3)
ax.set_title('B')
ax.imshow(b, cmap = "gray")
plt.show()

# print("min =", img.min())
# print("max =", img.max())