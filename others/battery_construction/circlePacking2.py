import numpy as np
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
