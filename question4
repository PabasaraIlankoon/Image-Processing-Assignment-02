import cv2 as cv
import numpy as np

img = cv.imread('crop_image.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

m, c = np.polyfit(x, y, 1)

theta_ls = np.degrees(np.arctan(m))
print("Estimated Angle (Least Squares):", theta_ls)