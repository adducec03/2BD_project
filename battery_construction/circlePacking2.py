import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
import json

def circle_packing_in_polygon(polygon_coords, cell_diameter):
    poly = Polygon(polygon_coords)
    r = cell_diameter / 2
    row_spacing = r * np.sqrt(3)
    col_spacing = cell_diameter

    minx, miny, maxx, maxy = poly.bounds
    packed = []
    y = miny + r
    row = 0

    while y + r <= maxy:
        x_offset = 0 if row % 2 == 0 else r
        x = minx + x_offset + r
        while x + r <= maxx:
            center = Point(x, y)
            circle = center.buffer(r, resolution=8)
            if poly.contains(circle):
                packed.append((x, y))
            x += col_spacing
        y += row_spacing
        row += 1

    return packed

def get_longest_edge_angle(polygon: Polygon):
    coords = list(polygon.exterior.coords)
    max_len = 0
    best_angle = 0
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length > max_len:
            max_len = length
            best_angle = np.degrees(np.arctan2(dy, dx))
    return best_angle

def find_best_packing(polygon: Polygon, radius: float):
    coords = list(polygon.exterior.coords)[:-1]  # Remove closing point
    original_centers = circle_packing_in_polygon(coords, radius * 2)
    
    angle = get_longest_edge_angle(polygon)
    rotated_polygon = rotate(polygon, -angle, origin='centroid')
    rotated_coords = list(rotated_polygon.exterior.coords)[:-1]
    rotated_centers = circle_packing_in_polygon(rotated_coords, radius * 2)

    if len(rotated_centers) > len(original_centers):
        centers = [rotate(Point(x, y), angle, origin=polygon.centroid) for x, y in rotated_centers]
        centers = [(p.x, p.y) for p in centers]
        best_angle = angle
    else:
        centers = original_centers
        best_angle = 0

    return centers, best_angle

#def plot_packing(polygon: Polygon, centers: list, radius: float, title="Packing Result"):
#    fig, ax = plt.subplots()
#    x, y = polygon.exterior.xy
#    ax.plot(x, y, color='black')
#
#    for cx, cy in centers:
#        circle = plt.Circle((cx, cy), radius, edgecolor='blue', facecolor='lightblue', alpha=0.6)
#        ax.add_patch(circle)
#
#    ax.set_aspect('equal')
#    ax.set_xlim(polygon.bounds[0] - radius, polygon.bounds[2] + radius)
#    ax.set_ylim(polygon.bounds[1] - radius, polygon.bounds[3] + radius)
#    plt.title(f"{title} - Cerchi totali: {len(centers)}")
#    plt.show()

if __name__ == "__main__":
    # Carica i punti
    with open("1.json", "r") as f:
        vertices = json.load(f)

    x, y = zip(*vertices)
    poly = Polygon(vertices)
    circle_radius = 9

    # Trova la miglior disposizione con il metodo adattivo
    best_centers, best_angle = find_best_packing(poly, circle_radius)

    # Visualizza il risultato
#    plot_packing(poly, best_centers, circle_radius, title=f"Disposizione ottimale (Angolo: {best_angle}°)")

    # Salva il risultato
    export_data = {
        "polygon": vertices,
        "circles": [
            {"center": [x, y], "radius": circle_radius}
            for x, y in best_centers
        ]
    }

    with open("polygon_with_circles.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Cerchi trovati: {len(best_centers)} | Rotazione migliore: {best_angle:.1f}°")