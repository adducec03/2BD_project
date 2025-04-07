import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely import affinity


def generate_hex_grid(polygon: Polygon, radius: float):
    minx, miny, maxx, maxy = polygon.bounds
    
    dx = 2 * radius
    dy = np.sqrt(3) * radius

    points = []
    y = miny
    row = 0
    while y <= maxy:
        x = minx + (radius if row % 2 else 0)
        while x <= maxx:
            center = Point(x, y)
            # Controlla se il cerchio centrato qui entra completamente nel poligono
            if polygon.contains(center.buffer(radius)):
                points.append((x, y))
            x += dx
        y += dy
        row += 1
    return points


def plot_packing(polygon: Polygon, centers: list, radius: float):
    fig, ax = plt.subplots()
    
    # Plot poligono
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='black')

    # Plot cerchi
    for cx, cy in centers:
        circle = plt.Circle((cx, cy), radius, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(polygon.bounds[0] - radius, polygon.bounds[2] + radius)
    ax.set_ylim(polygon.bounds[1] - radius, polygon.bounds[3] + radius)
    plt.title(f"Cerchi inseriti: {len(centers)}")
    plt.show()


if __name__ == "__main__":
    # Esempio di poligono concavo
    vertices = [
        (0, 0), (6, 0), (6, 2), (4, 2), (4, 4), (6, 4), (6, 6), (0, 6), (0, 4), (2, 4), (2, 2), (0, 2)
    ]
    poly = Polygon(vertices)

    # Raggio dei cerchi da impacchettare
    circle_radius = 0.4

    # Calcolo
    centers = generate_hex_grid(poly, circle_radius)

    # Visualizzazione
    plot_packing(poly, centers, circle_radius)

    print(f"Numero massimo di cerchi inseriti: {len(centers)}")
