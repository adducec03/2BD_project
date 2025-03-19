import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

# Controlla se un nuovo cerchio può essere aggiunto senza collisioni
def is_valid_circle(new_circle, circles, radius):
    min_distance = 2 * radius  # Nessuna sovrapposizione
    for c in circles:
        if new_circle.centroid.distance(c.centroid) < min_distance:
            return False
    return True

# Genera una disposizione iniziale a griglia esagonale
def generate_initial_solution(polygon, radius):
    circles = []
    min_x, min_y, max_x, max_y = polygon.bounds
    step_x = 2 * radius
    step_y = np.sqrt(3) * radius

    for i, x in enumerate(np.arange(min_x, max_x, step_x)):
        for j, y in enumerate(np.arange(min_y, max_y, step_y)):
            if i % 2 == 1:
                y += radius * np.sqrt(3) / 2  

            new_circle = Point(x, y).buffer(radius)
            if polygon.contains(new_circle) and is_valid_circle(new_circle, circles, radius):
                circles.append(new_circle)

    return circles

# Calcola l'area libera
def compute_free_area(polygon, circles):
    filled_area = sum(c.area for c in circles)
    return polygon.area - filled_area

# Genera una nuova configurazione spostando o aggiungendo cerchi validi
def perturb_solution(circles, polygon, radius):
    new_circles = circles.copy()
    step = radius / 2  # Spostamenti piccoli

    action = random.choice(["add", "move"])

    if action == "add":
        for _ in range(20):  
            x, y = np.random.uniform(polygon.bounds[0], polygon.bounds[2]), np.random.uniform(polygon.bounds[1], polygon.bounds[3])
            new_circle = Point(x, y).buffer(radius)
            if polygon.contains(new_circle) and is_valid_circle(new_circle, new_circles, radius):
                new_circles.append(new_circle)
                break

    elif action == "move" and new_circles:
        idx = random.randint(0, len(new_circles) - 1)
        old_circle = new_circles[idx]
        for _ in range(10):  
            x, y = old_circle.centroid.x + np.random.uniform(-step, step), old_circle.centroid.y + np.random.uniform(-step, step)
            new_circle = Point(x, y).buffer(radius)
            if polygon.contains(new_circle) and is_valid_circle(new_circle, new_circles[:idx] + new_circles[idx+1:], radius):
                new_circles[idx] = new_circle
                break

    return new_circles

# Algoritmo Simulated Annealing con stop automatico
def simulated_annealing(polygon, radius, initial_temp=1000, cooling_rate=0.9995, max_attempts=10000):
    current_solution = generate_initial_solution(polygon, radius)
    best_solution = current_solution
    current_area = compute_free_area(polygon, current_solution)
    best_area = current_area
    temp = initial_temp
    no_improvement = 0

    while no_improvement < 1000:  
        new_solution = perturb_solution(current_solution, polygon, radius)
        new_area = compute_free_area(polygon, new_solution)

        if new_area < best_area or np.exp((current_area - new_area) / temp) > random.random():
            current_solution = new_solution
            current_area = new_area
            if new_area < best_area:
                best_solution = new_solution
                best_area = new_area
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1

        temp *= cooling_rate

    return best_solution

# Funzione per visualizzare il risultato
def plot_solution(polygon, circles, radius):
    fig, ax = plt.subplots()
    
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label="Poligono")

    for circle in circles:
        centroid = circle.centroid
        patch = patches.Circle((centroid.x, centroid.y), radius, color='b', alpha=0.5)
        ax.add_patch(patch)

    ax.set_xlim(polygon.bounds[0] - 1, polygon.bounds[2] + 1)
    ax.set_ylim(polygon.bounds[1] - 1, polygon.bounds[3] + 1)
    ax.set_aspect('equal')
    plt.legend()
    plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    poly = Polygon([(0, 0), (10, 0), (12, 6), (5, 10), (-2, 6)])
    radius = 1  

    optimal_circles = simulated_annealing(poly, radius)

    plot_solution(poly, optimal_circles, radius)
