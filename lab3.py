

import numpy as np
import math
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
#  Utility Functions
# -----------------------------------------------------------------

def euclidean_distance_matrix(coords):
    """
    Compute pairwise Euclidean distances for a set of coordinates.
    
    This is a fast NumPy trick to build a distance matrix.
    It subtracts every coordinate from every other coordinate.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    # np.sum(diff**2, axis=-1) calculates the squared distance (dx^2 + dy^2)
    # np.sqrt() gives the final Euclidean distance
    return np.sqrt(np.sum(diff**2, axis=-1))

def tour_length(route, dist_matrix):
    """Compute total length of a tour."""
    n = len(route)
    
    # Sum the distance from each city 'i' to the next city in the route.
    # The (i+1) % n wraps around from the last city back to the first.
    return sum(dist_matrix[route[i], route[(i+1) % n]] for i in range(n))

def two_opt_swap(route):
    """
    Perform a 2-opt swap: reverse a subsequence of the tour.
    This is a common way to create a "neighbor" solution in TSP.
    """
    n = len(route)
    
    # 1. Pick two random, distinct indices
    a, b = sorted(random.sample(range(n), 2))
    
    # 2. Create the new route:
    #    - Part 1: route[0] up to route[a-1]
    #    - Part 2: route[a] up to route[b]... *but reversed*
    #    - Part 3: route[b+1] to the end
    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
    return new_route

# -----------------------------------------------------------------
#  Simulated Annealing Algorithm
# -----------------------------------------------------------------
def simulated_annealing(coords, T0=100, Tmin=1e-3, alpha=0.995, max_iter=20000):
    """
    Solves the TSP using Simulated Annealing.
    
    Parameters:
    - T0:       Initial "temperature". High temp = more likely to accept bad moves.
    - Tmin:     Final "temperature". The algorithm stops when T drops below this.
    - alpha:    "Cooling rate". Multiplied by T in each iteration (e.g., 0.995).
    - max_iter: Max iterations per temperature level (prevents getting stuck).
    """
    
    n = len(coords)
    dist_matrix = euclidean_distance_matrix(coords)

    # 1. Initialize with a random route
    route = list(range(n))
    random.shuffle(route)
    
    best_route = route[:]  # Make a copy
    best_cost = tour_length(route, dist_matrix)
    
    current_cost = best_cost
    
    T = T0
    history = [best_cost] # Store costs to plot later

    # Start the annealing loop
    while T > Tmin:
        for it in range(max_iter):
            
            # 2. Generate a neighbor solution
            candidate = two_opt_swap(route)
            candidate_cost = tour_length(candidate, dist_matrix)
            
            # 3. Calculate cost difference
            delta = candidate_cost - current_cost

            # 4. Acceptance Criterion (The core of SA)
            #    - If delta < 0: The new route is better. Always accept it.
            #    - If delta > 0: The new route is worse.
            #      We might *still* accept it with a probability of math.exp(-delta / T).
            #      This allows the algorithm to "escape" local minimums.
            if delta < 0 or random.random() < math.exp(-delta / T):
                route = candidate
                current_cost = candidate_cost
                
                # Check if this is the best-so-far solution
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = route[:] # Make a copy

        # Store history for plotting
        history.append(best_cost)
        
        # 5. Cool down
        T *= alpha

    return best_route, best_cost, history

# -----------------------------------------------------------------
#  Run the Algorithm
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    # --- Configuration ---
    NUM_CITIES = 5
    GRID_SIZE = 100
    
    # Generate random cities
    np.random.seed(42) # Use a seed for reproducible results
    coords = np.random.rand(NUM_CITIES, 2) * GRID_SIZE
    
    # --- Run ---
    print(f"Running Simulated Annealing for {NUM_CITIES} cities...")
    best_route, best_cost, history = simulated_annealing(coords)
    print(f"Best tour length: {best_cost:.2f}")

    # --- Plot 1: Progress (Cost over Time) ---
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration (x1000)")
    plt.ylabel("Best Tour Length")
    plt.title("Simulated Annealing Progress")
    plt.show()
    
    # --- Plot 2: Final Tour Map ---
    plt.figure(figsize=(6, 6))
    
    # Re-order coordinates based on the best route
    # We add best_route[0] to the end to close the loop for plotting
    ordered_coords = coords[best_route + [best_route[0]]]
    
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-') # 'o-' plots points and lines
    plt.title(f"Final Tour (Length = {best_cost:.2f})")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    

    
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def euclidean_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def tour_length(route, dist_matrix):
    n = len(route)
    return sum(dist_matrix[route[i], route[(i+1) % n]] for i in range(n))

def two_opt_swap(route):
    n = len(route)
    a, b = sorted(random.sample(range(n), 2))
    return route[:a] + route[a:b+1][::-1] + route[b+1:]

def simulated_annealing(coords, T0=100, Tmin=1e-3, alpha=0.995, max_iter=20000):
    n = len(coords)
    dist_matrix = euclidean_distance_matrix(coords)
    route = list(range(n))
    random.shuffle(route)
    best_route = route[:]
    best_cost = tour_length(route, dist_matrix)
    current_cost = best_cost
    T = T0
    history = [best_cost]

    while T > Tmin:
        for _ in range(max_iter):
            candidate = two_opt_swap(route)
            candidate_cost = tour_length(candidate, dist_matrix)
            delta = candidate_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                route = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = route[:]
        history.append(best_cost)
        T *= alpha
    return best_route, best_cost, history

if __name__ == "__main__":
    NUM_CITIES = 5
    GRID_SIZE = 100
    np.random.seed(42)
    coords = np.random.rand(NUM_CITIES, 2) * GRID_SIZE
    best_route, best_cost, history = simulated_annealing(coords)
    print(f"Best tour length: {best_cost:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration (x1000)")
    plt.ylabel("Best Tour Length")
    plt.title("Simulated Annealing Progress")
    plt.show()

    plt.figure(figsize=(6, 6))
    ordered_coords = coords[best_route + [best_route[0]]]
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-')
    plt.title(f"Final Tour (Length = {best_cost:.2f})")
    plt.show()

