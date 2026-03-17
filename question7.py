import cv2 as cv
import numpy as np

img = cv.imread('crop_image.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

data = np.vstack((x, y)).T
mean = np.mean(data, axis=0)
data_centered = data - mean

U, S, Vt = np.linalg.svd(data_centered)
direction = Vt[0]

dx, dy = direction

theta_tls = np.degrees(np.arctan2(dy, dx))
print("Estimated Angle (TLS):", theta_tls)