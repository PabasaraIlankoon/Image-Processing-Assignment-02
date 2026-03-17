import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('crop_image.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# Least Squares
m, c = np.polyfit(x, y, 1)

x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + c

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label='Points')
plt.plot(x_line, y_line, 'r', label='Least Squares')

plt.legend()
plt.gca().invert_yaxis()
plt.title("Least Squares Fit")

plt.show()