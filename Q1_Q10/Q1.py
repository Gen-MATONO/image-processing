import numpy as np
import matplotlib.pyplot as plt
from skimage import io

img_original = io.imread('https://raw.githubusercontent.com/Gen-MATONO/Gasyori100knock/master/Question_01_10/imori.jpg')

img = img_original.copy()
img = img[..., :: -1]

plt.figure()
ax = plt.subplot()
ax.imshow(img)
plt.show()

##############################################
# ANSWER
##############################################
import cv2

# function: BGR -> RGB
def BGR2RGB(img): # Cv2's colour order is BGR
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img

# Read image
img = cv2.imread("imori.jpg")  # There isn't imori.jpg in this directory. So you should alternatively use io from skimage.

# BGR -> RGB
img = BGR2RGB(img)

# Save result
cv2.imwrite("out.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0) # Program is going to stop until you type sny key.
cv2.destroyAllWindows() # Brake all windows