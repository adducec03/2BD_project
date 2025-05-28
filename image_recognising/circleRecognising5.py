import cv2
import numpy as np

output = cv2.imread('case1.jpg')
orig = cv2.imread('photos/1.jpg')

# convert to grayscale channels
b, g, r = cv2.split(orig)
processed = np.zeros_like(r)

def channel_processing(channel):
    channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 55, 7)
    channel = cv2.dilate(channel, None, iterations=1)
    channel = cv2.erode(channel, None, iterations=1)
    return channel

def inter_centre_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def colliding_circles(circles):
    if circles is None:
        return False
    for i, circle1 in enumerate(circles):
        for circle2 in circles[i+1:]:
            x1, y1, r1 = circle1
            x2, y2, r2 = circle2
            if inter_centre_distance(x1, y1, x2, y2) < r1 + r2:
                return True
    return False

def find_circles(processed, low):
    while True:
        circles = cv2.HoughCircles(processed,
                                    cv2.HOUGH_GRADIENT,
                                    dp=1.2,
                                    minDist=40,       # distanza minima tra centri
                                    param1=50,        # soglia Canny (bordo)
                                    param2=100,        # soglia per accettare un centro di cerchio
                                    minRadius=15,     # raggio minimo da rilevare
                                    maxRadius=60      # raggio massimo da rilevare
                                    )
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            print('number of circles:', len(circles))
            if not colliding_circles(circles):
                break
        low += 1
        print('trying with LOW =', low)
    return circles

def draw_circles(circles, output):
    if circles is not None:
        print(len(circles), 'circles found')
        for x, y, r in circles:
            cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
            cv2.circle(output, (x, y), r, (255, 0, 0), 3)

# process each channel
r = channel_processing(r)
g = channel_processing(g)
b = channel_processing(b)

# combine with logical AND
processed = cv2.bitwise_and(r, g)
processed = cv2.bitwise_and(processed, b)

cv2.imshow('before canny', processed)

# apply Canny + smoothing
processed = cv2.Canny(processed, 5, 70)
processed = cv2.GaussianBlur(processed, (7, 7), 0)

cv2.imshow('processed', processed)

# find and draw circles
circles = find_circles(processed, 100)
draw_circles(circles, output)

cv2.imshow('original with circles', output)
cv2.imwrite('case1_output.jpg', output)
cv2.waitKey(0)
cv2.destroyAllWindows()