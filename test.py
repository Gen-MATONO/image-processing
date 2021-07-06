import cv2
import numpy as np

# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# Otsu Binalization
def otsu_binarization(img):
    max_sigma = 0
    max_t = 0
    H, W = img.shape

    # determine threshold
    for _t in range(1, 256):
        # v0 and v1 are the array of pixels in class1 and class2, respectively.
        v0 = img[np.where(img < _t)] # np.where(cond)はcondを満たすときのindexをタプルで返す．
                                     # img[((x),(y))]は二重タプルによる，配列の座標指定，(x, y)のimgを返す．
        v1 = img[np.where(img >= _t)]

        # m0 and m1 are the average of each classes, respectively.
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        m1 = np.mean(v1) if len(v1) > 0 else 0.

        # w0 and w1 are the number of of each classes, respectively.
        # There are normalized by all pixels(H * W).
        w0 = len(v0) / (H * W)
        w1 = len(v1) / (H * W)

        # Sigma is distribution between two classes.
        sigma = w0 * w1 * ((m0 - m1) ** 2)

        # Otsu's algorithm determines the threshold when sigma is max value.
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    img[img < th] = 0
    img[img >= th] = 255

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float64)

# Grayscale
out = BGR2GRAY(img)

# Otsu's binarization
out = otsu_binarization(out)

# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()