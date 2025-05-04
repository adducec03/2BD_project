import json
import matplotlib.pyplot as plt

# === CONFIGURAZIONE ===
file_path = 'polygon_with_circles_and_connections.json'  # <-- cambia nome file se necessario

# === FUNZIONE DI VISUALIZZAZIONE ===
def visualizza_poligono_da_lista(percorso_file):
    try:
        with open(percorso_file, 'r') as f:
            polygon = json.load(f)

        if not isinstance(polygon, list) or not all(isinstance(p, list) and len(p) == 2 for p in polygon):
            raise ValueError("Il file JSON non contiene una lista valida di coordinate [x, y].")

        # Separiamo le coordinate in X e Y
        x, y = zip(*polygon)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='blue')
        plt.fill(x, y, alpha=0.2, color='skyblue')  # riempimento opzionale
        plt.title('Visualizzazione del Poligono')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    except Exception as e:
        print(f"Errore: {e}")

# === ESECUZIONE ===
if __name__ == "__main__":
    visualizza_poligono_da_lista(file_path)