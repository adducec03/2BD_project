import cv2
import numpy as np

# Carica l'immagine
img = cv2.imread("immagine.jpg")
output = img.copy()

# Converti in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applica un blur per ridurre il rumore (facilita il rilevamento dei cerchi)
gray_blurred = cv2.medianBlur(gray, 5)

# Applica il filtro di Canny (opzionale ma utile per visualizzare i bordi)
edges = cv2.Canny(gray_blurred, 50, 150)

# Rileva i cerchi con la trasformata di Hough
cerchi = cv2.HoughCircles(
    gray_blurred,                # Immagine di input (sfocata)
    cv2.HOUGH_GRADIENT,          # Metodo
    dp=1.0,                      # Inverso del rapporto di risoluzione
    minDist=80,                  # Distanza minima tra i centri
    param1=50,                   # Soglia per Canny (interna a Hough)
    param2=30,                   # Soglia accumulatore per il rilevamento
    minRadius=25,                # Raggio minimo
    maxRadius=60                 # Raggio massimo
)

# Se sono stati trovati cerchi
if cerchi is not None:
    cerchi = np.uint16(np.around(cerchi))
    for i in cerchi[0, :]:
        # Disegna il cerchio
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Disegna il centro del cerchio
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

# Mostra i risultati
cv2.imshow("Cerchi rilevati", output)
cv2.imshow("Bordi (Canny)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()