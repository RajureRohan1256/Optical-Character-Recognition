import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


path1 = '/Users/rohanrajure/Code/HTML/name.jpeg'
path2 = '/Users/rohanrajure/Code/PYTHON/Machine learning/OCR/main_menu.jpg'
# importing image
image = cv.imread(path1)


# converting image into grayscale
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

main_menu = '/Users/rohanrajure/Code/PYTHON/python-practice/venv/lib/python3.11/site-packages/cv2/data/main_menu.xml'
eye = '/Users/rohanrajure/Code/PYTHON/python-practice/venv/lib/python3.11/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml'

# now using XML file for detection
hotel_data = cv.CascadeClassifier(cv.data.haarcascades + main_menu)
found = hotel_data.detectMultiScale(grayscale, minSize=(20, 20))
amount_found = len(found)
if amount_found != 0:

    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:

        # We draw a green rectangle around
        # every recognized sign
        cv.rectangle(image, (x, y),
                     (x + height, y + width),
                     (0, 255, 0), 2)

# plt.subplot(1, 1, 1)
# plt.imshow(image)
# plt.show()

# displaying the Image
cv.imshow('hotel_main_menu', image)
cv.waitKey(0)
cv.destroyAllWindows()
