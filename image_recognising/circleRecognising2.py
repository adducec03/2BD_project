import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carica l'immagine
img = cv2.imread('photos/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sfoca per ridurre rumore
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Rileva i bordi
edges = cv2.Canny(blurred, 50, 150)

# Trova cerchi con HoughCircles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=300,
    param2=30,
    minRadius=30,
    maxRadius=100
)

# Disegna i cerchi trovati
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# Mostra risultato
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()