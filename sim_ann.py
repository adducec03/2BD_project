import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# Parametri della batteria e delle celle
cell_diameter = 18  # Diametro della cella (mm)
num_cells = 54  # Numero di celle

# Definisci la forma della batteria come un poligono qualsiasi (ad esempio un esagono)
battery_shape = Polygon([(0, 0), (200, 0), (220, 50), (200, 100), (0, 100), (-20, 50)])

# Funzione per calcolare la distanza tra due celle
def distance(cell1, cell2):
    return np.sqrt((cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2)

# Funzione per calcolare la capacità massima di celle che possono entrare nel poligono
def max_cells_in_polygon():
    max_x = max(battery_shape.bounds[2], battery_shape.bounds[0])  # Larghezza del poligono
    max_y = max(battery_shape.bounds[3], battery_shape.bounds[1])  # Altezza del poligono
    
    # Calcoliamo il numero massimo di celle che possiamo collocare in una disposizione regolare
    cells_per_row = int(max_x // cell_diameter)
    cells_per_column = int(max_y // cell_diameter)

    # Restituiamo il numero massimo di celle che possono entrare nella batteria
    return cells_per_row * cells_per_column

# Funzione per generare posizioni regolari all'interno della batteria
def generate_regular_positions():
    positions = []
    max_x = max(battery_shape.bounds[2], battery_shape.bounds[0])  # Larghezza del poligono
    max_y = max(battery_shape.bounds[3], battery_shape.bounds[1])  # Altezza del poligono
    
    # Calcoliamo il numero massimo di celle che possiamo collocare in un modello regolare
    cells_per_row = int(max_x // cell_diameter)
    cells_per_column = int(max_y // cell_diameter)
    
    for row in range(cells_per_column):
        for col in range(cells_per_row):
            x = col * cell_diameter + cell_diameter / 2
            y = row * cell_diameter + cell_diameter / 2
            point = Point(x, y)

            # Verifica che la cella è dentro la forma poligonale e non collide con altre celle
            if is_valid_position(positions, [x, y]):
                positions.append([x, y])
                # Verifica se abbiamo raggiunto il numero di celle richiesto
                if len(positions) >= num_cells:
                    return positions

    # Se non raggiungiamo il numero di celle richiesto in una disposizione regolare, continua
    # in modo che il Simulated Annealing possa cercare una soluzione ottimale
    return positions

# Funzione per verificare che una cella sia valida (tutto il bordo della circonferenza è dentro la batteria)
def is_valid_position(positions, new_position):
    # Verifica che la cella sia dentro la batteria
    directions = [(np.cos(np.pi/4 * i), np.sin(np.pi/4 * i)) for i in range(8)]
    for dx, dy in directions:
        edge_point = Point(new_position[0] + dx * cell_diameter / 2, new_position[1] + dy * cell_diameter / 2)
        if not battery_shape.contains(edge_point):
            return False
    
    # Verifica che non ci sia sovrapposizione con altre celle
    for pos in positions:
        if distance(pos, new_position) < cell_diameter:
            return False
    
    return True

# Funzione per generare una posizione casuale all'interno della batteria
def generate_random_position():
    # Genera una posizione casuale all'interno della batteria
    x = random.uniform(battery_shape.bounds[0] + cell_diameter / 2, battery_shape.bounds[2] - cell_diameter / 2)
    y = random.uniform(battery_shape.bounds[1] + cell_diameter / 2, battery_shape.bounds[3] - cell_diameter / 2)
    point = Point(x, y)
    if battery_shape.contains(point):
        return [x, y]
    return generate_random_position()  # Ricomincia se la posizione non è valida

# Funzione di Simulated Annealing per ottimizzare la disposizione delle celle
def simulated_annealing(max_iter=10000, max_attempts=1000):
    # Verifica se il numero di celle richieste è inferiore alla capacità massima del poligono
    max_cells = max_cells_in_polygon()
    if num_cells > max_cells:
        print(f"Avviso: Non è possibile inserire {num_cells} celle nel poligono. La capacità massima è {max_cells}.")
        return []

    # Inizializza le posizioni regolari
    current_positions = generate_regular_positions()

    # Controlla che siano state generate almeno 'num_cells' celle, se no esegui il processo di Simulated Annealing
    while len(current_positions) < num_cells:
        # Aggiungi una cella alla volta in modo casuale e cerca una disposizione migliore
        new_pos = generate_random_position()
        if is_valid_position(current_positions, new_pos):
            current_positions.append(new_pos)

    # Funzione per calcolare il costo (obiettivo): la distanza totale tra le celle
    def calculate_cost(positions):
        total_cost = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_cost += distance(positions[i], positions[j])
        return total_cost

    # Parametri per il Simulated Annealing
    temperature = 1000
    cooling_rate = 0.99
    iterations = 0
    attempts = 0
    current_cost = calculate_cost(current_positions)
    best_positions = current_positions
    best_cost = current_cost

    # Simulated Annealing loop
    while iterations < max_iter and attempts < max_attempts:
        # Genera una nuova posizione tramite piccole modifiche (ad esempio spostando una cella)
        new_positions = current_positions.copy()
        index_to_modify = random.randint(0, len(current_positions) - 1)
        new_positions[index_to_modify] = generate_random_position()
        
        # Verifica che la nuova posizione sia valida
        if is_valid_position(new_positions, new_positions[index_to_modify]):
            new_cost = calculate_cost(new_positions)
            cost_difference = current_cost - new_cost
            
            # Decidi se accettare la nuova configurazione
            if cost_difference > 0 or random.uniform(0, 1) < np.exp(cost_difference / temperature):
                current_positions = new_positions
                current_cost = new_cost

                if new_cost < best_cost:
                    best_positions = new_positions
                    best_cost = new_cost
            attempts = 0  # Reset tentativi quando facciamo un progresso
        else:
            attempts += 1  # Incrementa il contatore di tentativi

        # Raffredda la temperatura
        temperature *= cooling_rate
        iterations += 1

    return best_positions

# Genera la disposizione ottimale delle celle (se possibile)
optimal_positions = simulated_annealing()

# Visualizza la disposizione delle celle come cerchi
if optimal_positions:
    plt.figure(figsize=(8, 8))
    for pos in optimal_positions:
        circle = plt.Circle((pos[0], pos[1]), cell_diameter / 2, edgecolor='blue', facecolor='none', lw=2)
        plt.gca().add_artist(circle)

    # Disegna la batteria come poligono
    x, y = battery_shape.exterior.xy
    plt.fill(x, y, alpha=0.3, color='gray')  # Colore della batteria
    plt.plot(x, y, color='black', linewidth=2)  # Bordo della batteria

    # Imposta i limiti della visualizzazione
    plt.xlim(0, battery_shape.bounds[2] + 10)
    plt.ylim(0, battery_shape.bounds[3] + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Ottimizzazione della disposizione delle celle in una batteria poligonale')
    plt.show()
