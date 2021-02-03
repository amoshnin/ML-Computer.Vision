import numpy as np
import cv2


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def rotate(image, angle, center=None, scale=1.0):
    (height, width) = image.shape[:2]
    if center is None: 
        center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (width, height))


def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)

    else: 
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation)
