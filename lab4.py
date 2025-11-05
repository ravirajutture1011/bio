import numpy as np
import random

# Distance matrix (4 cities)
distances = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

n = distances.shape[0]  # Number of cities

# Parameters
alpha = 1        # Pheromone influence
beta = 2         # Heuristic (distance) influence
rho = 0.1        # Evaporation rate
Q = 100          # Pheromone deposit amount
n_ants = 4       # Number of ants
n_iterations = 20  # Number of iterations

# Initialize pheromone matrix
pheromone = np.ones((n, n))

# --- Helper Function (from image_422b46.png) ---

def tour_length(tour):
    """Calculates the total length of a given tour."""
    # The %n handles the wrap-around from the last city back to the first
    return sum(distances[tour[i], tour[(i + 1) % n]] for i in range(n))

# --- Main Algorithm Loop (from image_422e0d.jpg and image_422e6c.png) ---

best_tour = None
best_len = float('inf')

# Main iteration loop
for it in range(n_iterations):
    all_tours = []
    
    # Ant loop: each ant builds a complete tour
    for k in range(n_ants):
        # Start each ant at a random city
        start = random.randint(0, n - 1)
        tour = [start]
        visited = {start}
        
        # Build the rest of the tour
        while len(tour) < n:
            i = tour[-1]  # Current city
            probs = []
            
            # Calculate probabilities to move to each unvisited city 'j'
            for j in range(n):
                if j not in visited:
                    # (pheromone^alpha) * (heuristic^beta)
                    # Heuristic (eta) is 1 / distance
                    tau = pheromone[i, j] ** alpha
                    eta = (1.0 / (distances[i, j] + 1e-12)) ** beta # +1e-12 to avoid division by zero
                    probs.append((j, tau * eta))
            
            # --- Probabilistic city selection ---
            total_prob = sum(val for _, val in probs)
            r = random.random() * total_prob
            cum_prob = 0
            
            for j, val in probs:
                cum_prob += val
                if r <= cum_prob:
                    tour.append(j)
                    visited.add(j)
                    break
                    
        all_tours.append(tour)
        
    # --- Pheromone Update (from image_422e6c.png) ---
    
    # 1. Evaporation
    pheromone *= (1 - rho)
    
    # 2. Deposit
    for tour in all_tours:
        L = tour_length(tour)
        
        # Update the best tour found so far
        if L < best_len:
            best_tour, best_len = tour, L
        
        # Deposit pheromone on the edges of this tour
        deposit = Q / L
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit  # Assuming a symmetric problem
            
    print(f"Iter {it + 1}: best length = {best_len}")

# --- Final Result ---
print("---")
print("Best tour:", best_tour)
print("Length =", best_len)



'''
import numpy as np
import random

distances = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

n = distances.shape[0]

alpha = 1
beta = 2
rho = 0.1
Q = 100
n_ants = 4
n_iterations = 20

pheromone = np.ones((n, n))

def tour_length(tour):
    return sum(distances[tour[i], tour[(i + 1) % n]] for i in range(n))

best_tour = None
best_len = float('inf')

for it in range(n_iterations):
    all_tours = []
    
    for k in range(n_ants):
        start = random.randint(0, n - 1)
        tour = [start]
        visited = {start}
        
        while len(tour) < n:
            i = tour[-1]
            probs = []
            
            for j in range(n):
                if j not in visited:
                    tau = pheromone[i, j] ** alpha
                    eta = (1.0 / (distances[i, j] + 1e-12)) ** beta
                    probs.append((j, tau * eta))
            
            total_prob = sum(val for _, val in probs)
            r = random.random() * total_prob
            cum_prob = 0
            
            for j, val in probs:
                cum_prob += val
                if r <= cum_prob:
                    tour.append(j)
                    visited.add(j)
                    break
                    
        all_tours.append(tour)
        
    pheromone *= (1 - rho)
    
    for tour in all_tours:
        L = tour_length(tour)
        
        if L < best_len:
            best_tour, best_len = tour, L
        
        deposit = Q / L
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit
            
    print(f"Iter {it + 1}: best length = {best_len}")

print("---")
print("Best tour:", best_tour)
print("Length =", best_len)

'''