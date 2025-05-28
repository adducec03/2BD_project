import cv2
import numpy as np

# Carica l'immagine
img = cv2.imread("photos/5.jpg")
output = img.copy()

# Converti in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applica un blur per ridurre il rumore (facilita il rilevamento dei cerchi)
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
gray_blurred = cv2.GaussianBlur(gray_blurred, (9, 9), 2)
gray_blurred = cv2.GaussianBlur(gray_blurred, (9, 9), 2)
gray_blurred = cv2.GaussianBlur(gray_blurred, (9, 9), 2)

# Applica il filtro di Canny (opzionale ma utile per visualizzare i bordi)
edges = cv2.Canny(gray_blurred, 30, 90)

# Rileva i cerchi con la trasformata di Hough
cerchi = cv2.HoughCircles(
    edges,                # Immagine di input (sfocata)
    cv2.HOUGH_GRADIENT,          # Metodo
    dp=1,                      # Inverso del rapporto di risoluzione
    minDist=120,                  # Distanza minima tra i centri
    param1=100,                   # Soglia per Canny (interna a Hough)
    param2=20,                   # Soglia accumulatore per il rilevamento
    minRadius=10,                # Raggio minimo
    maxRadius=90                 # Raggio massimo
)

# Se sono stati trovati cerchi
if cerchi is not None:
    cerchi = np.uint16(np.around(cerchi))
    for i in cerchi[0, :]:
        center = (i[0], i[1])
        radius = i[2]

        # Classificazione del polo
        if radius < 60:
            label = "Positivo"
            color = (0, 255, 0)  # Verde
        else:
            label = "Negativo"
            color = (0, 0, 255)  # Rosso

        # Disegna il cerchio e l'etichetta
        cv2.circle(output, center, radius, color, 3)
        cv2.putText(output, label, (i[0] - 20, i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Mostra i risultati
cv2.imshow("Cerchi rilevati", output)
cv2.imshow("Bordi (Canny)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()