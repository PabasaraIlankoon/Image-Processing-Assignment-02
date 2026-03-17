import cv2 as cv
import numpy as np
from sklearn.linear_model import RANSACRegressor

img = cv.imread('crop_image.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

x_reshaped = x.reshape(-1,1)

ransac = RANSACRegressor()
ransac.fit(x_reshaped, y)

m_ransac = ransac.estimator_.coef_[0]

theta_ransac = np.degrees(np.arctan(m_ransac))
print("Estimated Angle (RANSAC):", theta_ransac)