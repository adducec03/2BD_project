import cv2
import numpy as np

# Carica l'immagine
img = cv2.imread("photos/5.jpg")
output = img.copy()

# Pre-elaborazione: Scala di grigi ed equalizzazione dell'istogramma
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Filtro bilaterale per ridurre il rumore mantenendo i bordi
gray = cv2.bilateralFilter(gray, 9, 75, 75)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
# Rilevazione dei bordi
edges = cv2.Canny(gray, 50, 150)

# Morfologia per chiudere piccoli buchi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Rileva i cerchi con Hough
cerchi = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    param1=50, param2=30, minRadius=30, maxRadius=80
)

def classify_polarity(roi):
    """ Rileva il polo basato sul cerchio interno """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=20, minRadius=50, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda x: x[2])
        if largest_circle[2] > 80:
            return "Negativo", (0, 0, 255)  # Rosso
        else:
            return "Positivo", (0, 255, 0)  # Verde
    return "Sconosciuto", (0, 0, 0)  # Blu

# Controllo per evitare cerchi sovrapposti
detected_centers = []

if cerchi is not None:
    cerchi = np.uint16(np.around(cerchi))
    for c in cerchi[0, :]:
        center = (c[0], c[1])
        radius = c[2]

        # Evita cerchi sovrapposti
        collision = False
        for dc in detected_centers:
            if np.linalg.norm(np.array(center) - np.array(dc)) < radius:
                collision = True
                break
        
        if collision:
            continue

        detected_centers.append(center)

        # Estrai la regione di interesse (ROI) attorno al cerchio
        x, y, r = c[0], c[1], c[2]
        roi = output[y - r:y + r, x - r:x + r]

        # Classificazione del polo
        label, color = classify_polarity(roi)

        # Disegna il cerchio e l'etichetta
        cv2.circle(output, center, radius, color, 3)
        cv2.putText(output, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Mostra i risultati
cv2.imshow("Rilevamento Batterie", output)
cv2.imshow("Bordi (Canny)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()