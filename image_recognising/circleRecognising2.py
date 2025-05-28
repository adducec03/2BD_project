import cv2
import numpy as np

# Carica l'immagine
img = cv2.imread("photos/1.jpg")
output = img.copy()

# Pre-elaborazione: Scala di grigi ed equalizzazione dell'istogramma
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Filtro bilaterale per ridurre il rumore mantenendo i bordi
gray = cv2.bilateralFilter(gray, 9, 75, 75)

# Soglia binaria inversa
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Morfologia per chiudere piccoli buchi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Trova i contorni
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def classify_polarity(roi):
    """ Rileva il polo basato sul cerchio interno """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Trova cerchi interni nella ROI
    circles = cv2.HoughCircles(roi_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda x: x[2])
        if largest_circle[2] > 15:
            return "Negativo", (0, 0, 255)  # Rosso
        else:
            return "Positivo", (0, 255, 0)  # Verde
    return "Sconosciuto", (255, 0, 0)  # Blu

# Analizza i contorni per trovare le batterie
# Analizza i contorni per trovare le batterie
for contour in contours:
    # Filtra i contorni piccoli
    area = cv2.contourArea(contour)
    if area < 500:
        continue

    # Approssima il contorno a un cerchio
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    # Filtro per evitare cerchi troppo piccoli o grandi
    if radius < 20 or radius > 90:
        continue

    # Calcola i limiti della ROI garantendo che siano nell'immagine
    x1 = max(0, int(x - radius))
    y1 = max(0, int(y - radius))
    x2 = min(output.shape[1], int(x + radius))
    y2 = min(output.shape[0], int(y + radius))

    # Estrai la ROI della batteria
    roi = output[y1:y2, x1:x2]

    # Classificazione del polo
    label, color = classify_polarity(roi)

    # Disegna il cerchio e l'etichetta
    cv2.circle(output, center, radius, color, 3)
    cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Mostra i risultati
cv2.imshow("Rilevamento Batterie", output)
cv2.imshow("Immagine Binaria", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()