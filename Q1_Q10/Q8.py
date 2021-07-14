# ##############################################################################
# Q8 : Max Pooling
# In previous question, I used average of each areas for pooling proccess.
# In contrast, this question uses a max value of each areas. This method is
# named 'Max Pooling'.
# ##############################################################################


# ******************************************************************************
# ------------------------------------------------------------------------------
# my code
# ------------------------------------------------------------------------------
import numpy as np
import cv2

def max_pooling(img, x):
    out = img
    h, w, d = out.shape

    for i in range(0, h, x):
        for j in range(0, w, x):
            for k in range(0, d):
                out[i:i+x, j:j+x, k] = np.max(out[i:i+x, j:j+x, k])

    return out.astype(np.uint8)

# Read image
img_original = cv2.imread('imori.jpg')
img = img_original.copy()

# Read area for pooling
# x = int(input('Input range used for pooling process.(What is x for x**x ?)'))
x = 8

# Call function for image
img = max_pooling(img, x)

cv2.imshow('imori',img)
cv2.waitKey(0)
cv2.destroyAllWindows
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------
# answer code
# ------------------------------------------------------------------------------
import cv2
import numpy as np

# max pooling
def max_pooling(img, G=8):
    # Max Pooling
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])

    return out


# Read image
img = cv2.imread("imori.jpg")

# Max pooling
out = max_pooling(img)

# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ******************************************************************************