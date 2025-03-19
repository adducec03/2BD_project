import numpy as np
import random
import matplotlib.pyplot as plt

# Parametri della batteria e delle celle
battery_width = 200  # Larghezza della batteria (mm)
battery_length = 50  # Lunghezza della batteria (mm)
cell_diameter = 18  # Diametro della cella (mm)
num_cells = 22  # Numero di celle

# Funzione per calcolare la distanza tra due celle
def distance(cell1, cell2):
    return np.sqrt((cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2)

# Funzione per generare posizioni regolari
def generate_regular_positions():
    # Calcoliamo quante celle possono entrare in ogni direzione
    cells_per_row = int(battery_width // cell_diameter)
    cells_per_column = int(battery_length // cell_diameter)
    
    # Verifica se il numero di celle richiesto è maggiore di quelle che entrano nel rettangolo
    if num_cells > cells_per_row * cells_per_column:
        print(f"Impossibile posizionare {num_cells} celle in un rettangolo {battery_width}x{battery_length}!")
        return []
    
    positions = []
    for row in range(cells_per_column):
        for col in range(cells_per_row):
            if len(positions) < num_cells:
                x = col * cell_diameter + cell_diameter / 2
                y = row * cell_diameter + cell_diameter / 2
                positions.append([x, y])
            else:
                break
    return positions

# Funzione per generare una posizione casuale all'interno del rettangolo
def generate_random_position():
    x = random.uniform(cell_diameter / 2, battery_width - cell_diameter / 2)
    y = random.uniform(cell_diameter / 2, battery_length - cell_diameter / 2)
    return [x, y]

# Funzione per controllare se una cella si sovrappone con altre celle
def is_valid_position(positions, new_position):
    for pos in positions:
        if distance(pos, new_position) < cell_diameter:  # Se la distanza tra le celle è inferiore al diametro, la posizione non è valida
            return False
    return True

# Funzione di Simulated Annealing per ottimizzare la disposizione delle celle
def simulated_annealing(max_iter=10000, max_attempts=1000):
    current_positions = generate_regular_positions()
    if not current_positions:
        return []
    


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

    # Imposta i limiti della visualizzazione
    plt.xlim(0, battery_width)
    plt.ylim(0, battery_length)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Ottimizzazione della disposizione delle celle con Simulated Annealing')
    plt.show()
