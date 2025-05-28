import cv2
import numpy as np

def detect_circles(image_gray):
    # Blur per ridurre rumore
    blurred = cv2.GaussianBlur(image_gray, (9, 9), 2)

    # Canny + HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,         # distanza tra centri di batterie (regolare)
        param1=200,         # soglia Canny
        param2=28,          # soglia centro cerchio (aumenta se rileva troppo)
        minRadius=50,       # batteria piccola
        maxRadius=100        # batteria grande
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return circles
    return []

def group_circles_by_row(circles, tolerance=15):
    rows = []
    for circle in sorted(circles, key=lambda c: c[1]):  # ordina per y (riga)
        added = False
        for row in rows:
            if abs(circle[1] - row[0][1]) < tolerance:
                row.append(circle)
                added = True
                break
        if not added:
            rows.append([circle])
    # ordina ogni riga per x
    rows = [sorted(row, key=lambda c: c[0]) for row in rows]
    return rows

def draw_detected_rows(image, grouped_circles):
    for row in grouped_circles:
        for (x, y, r) in row:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    # opzionale: disegna linee tra i centri della stessa riga
    for row in grouped_circles:
        for i in range(len(row) - 1):
            pt1 = (row[i][0], row[i][1])
            pt2 = (row[i+1][0], row[i+1][1])
            cv2.line(image, pt1, pt2, (255, 0, 0), 1)

def main():
    image = cv2.imread('photos/1.jpg')
    if image is None:
        print("Errore: immagine non trovata.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = detect_circles(gray)
    
    print(f"Rilevati {len(circles)} cerchi")
    
    grouped = group_circles_by_row(circles)
    
    print(f"Raggruppati in {len(grouped)} righe")
    for i, row in enumerate(grouped):
        print(f"  Riga {i+1}: {len(row)} cerchi")

    draw_detected_rows(image, grouped)
    
    cv2.imshow("Batterie rilevate", image)
    cv2.imwrite("batterie_output.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()