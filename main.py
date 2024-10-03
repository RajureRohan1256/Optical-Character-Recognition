# Documentation ⬇️
# Note:- This is Computer Vision Project (machine learning).
# This project is for detecting the letter which is hand written like an 'Image' etc.

# importing required libraries
import numpy as np
import cv2 as cv

# importing required images
path = '/Users/rohanrajure/Code/PYTHON/Computer Vision/Text_Recognition(Project)/HWLay.jpeg'
img = cv.imread(path)


# resizing those images
def rescale_image(frame, scale=.25):
    width = int(frame.shape[1] / scale)
    height = int(frame.shape[0] / scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_frame = rescale_image(img, scale=.25)

# bluring these image
gaussian_blur = cv.GaussianBlur(resized_frame, (5, 5), 0)
media_blur = cv.medianBlur(resized_frame, 5)


# converting images into grayscale
grayscale_gaussian = cv.cvtColor(gaussian_blur, cv.COLOR_BGR2GRAY)
grayscale_median = cv.cvtColor(media_blur, cv.COLOR_BGR2GRAY)

# threshold of images
threshold_1 = cv.threshold(grayscale_median, 0, 255,
                           cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

threshold_2 = cv.threshold(grayscale_gaussian, 0, 255,
                           cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]


# applying monopology
kernal1 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
# kernal1 = np.ones((5, 5), np.uint8)
opeing_img = cv.morphologyEx(threshold_2, cv.MORPH_OPEN, kernal1, iterations=1)
# closing_img = cv.morphologyEx(threshold_2, cv.MORPH_ERODE, kernal, iterations=1)

# inversing the opening_img
invert_opening_img = cv.threshold(
    opeing_img, 0, 255, cv.THRESH_BINARY)[1]


# showing an image into the terminal
# cv.imshow('Original_Image', resized_frame)
# cv.imshow('Gausian_blur', gaussian_blur)
# cv.imshow('Grayscale_gaussian', grayscale_gaussian)
# cv.imshow('Grayscale_median', grayscale_median)
# cv.imshow('threshold_otsu', threshold_1)
# cv.imshow('threshold_otsu1', threshold_2)
# cv.imshow('opened_img', opeing_img)
cv.imshow('invert_opening_img', invert_opening_img)
# cv.imshow('closing_img', closing_img)
cv.waitKey(0)
cv.destroyAllWindows()
