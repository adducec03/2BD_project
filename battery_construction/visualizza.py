import json
import matplotlib.pyplot as plt

# Percorso del file JSON
file_path = "polygon_2.json"

# Carica i vertici dal file JSON
with open(file_path, 'r') as f:
    vertices = json.load(f)

# Estrai le coordinate X e Y
x_coords, y_coords = zip(*vertices)

# Visualizza il poligono
plt.figure(figsize=(6, 6))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
plt.fill(x_coords, y_coords, alpha=0.2, color='skyblue')

# Etichette e opzioni grafiche
plt.title("Visualizzazione del Poligono da JSON")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')  # Proporzioni corrette
plt.show()