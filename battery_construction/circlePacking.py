import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely import affinity
from shapely.ops import unary_union
import json


def generate_hex_grid(polygon: Polygon, radius: float, angle: float = 0):
    # Applica rotazione al poligono
    rotated_polygon = affinity.rotate(polygon, angle, origin='centroid')
    
    minx, miny, maxx, maxy = rotated_polygon.bounds

    dx = 2 * radius
    dy = np.sqrt(3) * radius

    points = []
    y = miny
    row = 0
    while y <= maxy:
        x = minx + (radius if row % 2 else 0)
        while x <= maxx:
            center = Point(x, y)
            if rotated_polygon.contains(center.buffer(radius)):
                points.append((x, y))
            x += dx
        y += dy
        row += 1

    # Ruota indietro i punti
    rotated_back = [affinity.rotate(Point(x, y), -angle, origin=polygon.centroid) for x, y in points]
    return [(p.x, p.y) for p in rotated_back]

def generate_aligned_hex_grid(polygon: Polygon, radius: float, margin=3):
    """Genera una griglia esagonale allineata per rettangoli/quadrati partendo dal bordo, con margine fisso."""
    # Trova il rettangolo ruotato minimo per considerare i rettangoli inclinati
    oriented_bbox = polygon.minimum_rotated_rectangle
    bbox_coords = list(oriented_bbox.exterior.coords)[:4]

    # Calcolo dell'angolo di rotazione del bounding box rispetto agli assi
    v1 = np.array(bbox_coords[1]) - np.array(bbox_coords[0])
    angle = np.degrees(np.arctan2(v1[1], v1[0]))

    # Ruota il poligono per allinearlo con gli assi
    aligned_polygon = affinity.rotate(polygon, -angle, origin='centroid')

    minx, miny, maxx, maxy = aligned_polygon.bounds
    dx = 2 * radius
    dy = np.sqrt(3) * radius

    # Calcolo del margine in coordinate
    margin_x = (maxx - minx) * (margin / 1000)  # Approssimazione per il margine
    margin_y = (maxy - miny) * (margin / 1000)

    points = []
    y = miny + radius + margin_y  # Partenza dal bordo inferiore con margine
    row = 0

    while y <= maxy - radius - margin_y:
        # Allineamento alla colonna sinistra con margine
        x = minx + (radius if row % 2 else 0) + margin_x
        while x <= maxx - radius - margin_x:
            center = Point(x, y)
            if aligned_polygon.contains(center.buffer(radius)):
                points.append((x, y))
            x += dx
        y += dy
        row += 1

    # Ruota indietro i punti per tornare alla posizione originale
    aligned_back = [affinity.rotate(Point(x, y), angle, origin=polygon.centroid) for x, y in points]
    return [(p.x, p.y) for p in aligned_back]


#metodo che controlla se un nuovo cerchio può essere aggiunto senza collisioni
def add_extra_circles(polygon: Polygon, existing_centers: list, radius: float, spacing: float = 0.5):
    minx, miny, maxx, maxy = polygon.bounds
    new_points = []

    # Crea i buffer dei cerchi esistenti per confronto geometrico
    existing_buffers = [Point(x, y).buffer(radius) for x, y in existing_centers]
    all_buffers = unary_union(existing_buffers)  # Unione dei cerchi già piazzati

    dx = spacing * radius
    dy = spacing * radius

    x_vals = np.arange(minx, maxx + dx, dx)
    y_vals = np.arange(miny, maxy + dy, dy)

    for x in x_vals:
        for y in y_vals:
            center = Point(x, y)
            buffer = center.buffer(radius)

            if not polygon.contains(buffer):
                continue

            # Controllo preciso: il nuovo cerchio non deve intersecare altri
            if all_buffers.intersects(buffer):
                continue

            # Aggiungilo e aggiorna unione
            new_points.append((x, y))
            all_buffers = unary_union([all_buffers, buffer])

    return new_points


def plot_packing(polygon: Polygon, centers: list, radius: float, title="Packing"):
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='black')

    for cx, cy in centers:
        circle = plt.Circle((cx, cy), radius, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(polygon.bounds[0] - radius, polygon.bounds[2] + radius)
    ax.set_ylim(polygon.bounds[1] - radius, polygon.bounds[3] + radius)
    plt.title(f"{title} - Cerchi totali: {len(centers)}")
    plt.show()


#metodo che trova la rotazione ottimale per il packing
def find_best_rotation(polygon: Polygon, radius: float, angles=None):
    if angles is None:
        angles = np.arange(0, 62, 1)  # Rotazioni da 0° a 55° ogni 5°

    best_centers = []
    best_angle = 0

    for angle in angles:
        base_centers = generate_hex_grid(polygon, radius, angle=angle)
        extra_centers = add_extra_circles(polygon, base_centers, radius)
        total = base_centers + extra_centers

        #print(f"Rotazione {angle}° -> Totale cerchi: {len(total)}")
        if len(total) > len(best_centers):
            best_centers = total
            best_angle = angle

    return best_centers, best_angle

def is_rectangle_or_square(polygon: Polygon, angle_tolerance=5.0):
    """Verifica se il poligono è un rettangolo o quadrato basato sugli angoli interni."""
    coords = list(polygon.exterior.coords)
    if len(coords) != 5:  # Un rettangolo ha 4 lati (più uno chiuso)
        return False

    angles = []
    for i in range(4):
        v1 = np.array(coords[i + 1]) - np.array(coords[i])
        v2 = np.array(coords[(i + 2) % 4]) - np.array(coords[i + 1])
        
        # Calcola l'angolo tra i vettori
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.degrees(np.arccos(dot_product / norm_product))
        angles.append(angle)

    # Verifica che tutti gli angoli siano circa 90°
    if all(np.isclose(angle, 90, atol=angle_tolerance) for angle in angles):
        return True
    return False

def find_best_packing(polygon: Polygon, radius: float):
    """Sceglie il miglior metodo di packing a seconda della forma."""
    if is_rectangle_or_square(polygon):
        #print("🟦 Rilevato rettangolo/quadrato: uso griglia esagonale allineata al bordo.")
        best_centers = generate_aligned_hex_grid(polygon, radius)
        best_angle = 0
    else:
        #print("🔄 Forma irregolare: uso disposizione esagonale con rotazione ottimizzata.")
        best_centers, best_angle = find_best_rotation(polygon, radius)
    
    return best_centers, best_angle


if __name__ == "__main__":
    # Carica i punti
    with open("polygon.json", "r") as f:
        vertices = json.load(f)

    x, y = zip(*vertices)
    poly = Polygon(vertices)

    circle_radius = 9

    # Trova la miglior disposizione con il metodo adattivo
    best_centers, best_angle = find_best_packing(poly, circle_radius)

    # Visualizza il risultato
    #plot_packing(poly, best_centers, circle_radius, title=f"Disposizione ottimale (Angolo: {best_angle}°)")
    #print(f"Rotazione migliore: {best_angle}°, Totale cerchi: {len(best_centers)}")

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

    #print("✅ Risultato esportato in 'polygon_with_circles.json'")