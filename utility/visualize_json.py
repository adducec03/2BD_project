import json
import matplotlib.pyplot as plt

# Carica i dati JSON da un file
with open('polygon_with_circles.json', 'r') as file:
    data = json.load(file)

# Estrai il poligono e i cerchi
polygon = data['polygon']
circles = data['circles']

# Imposta il grafico
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.title('Visualizzazione Poligono e Cerchi')

# Disegna il poligono
polygon_x, polygon_y = zip(*polygon)
plt.plot(polygon_x, polygon_y, marker='o', linestyle='-', color='blue', label='Poligono')

# Disegna i cerchi
for circle in circles:
    center = circle['center']
    radius = circle['radius']
    circle_patch = plt.Circle(center, radius, color='red', alpha=0.5)
    ax.add_patch(circle_patch)

# Imposta i limiti degli assi
ax.set_xlim(0, 800)
ax.set_ylim(0, 600)

# Mostra la legenda e il grafico
plt.legend()
plt.show()
