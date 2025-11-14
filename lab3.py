import numpy as np
import math
import random
import matplotlib.pyplot as plt
 
def euclidean_distance_matrix(coords):
     n = len(coords)
     distance_maxtrix = np.zeros((n,n))
     for i in range(n):
         for j in range(n):
             city_i = coords[i]
             city_j = coords[j]
             
             dx = city_i[0] - city_j[0]
             dy = city_i[1] - city_j[1]

             distance = math.sqrt(dx**2 + dy**2)
             distance_maxtrix[i,j] = distance
             
     return distance_maxtrix


def tour_length(route, dist_matrix):
    n = len(route)
    return sum(dist_matrix[route[i], route[(i+1) % n]] for i in range(n))

def two_opt_swap(route):
    n = len(route)
    a, b = sorted(random.sample(range(n), 2))
    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
    return new_route


def simulated_annealing(coords, T0=100, Tmin=1e-3, alpha=0.995, max_iter=20000):
  
    n = len(coords)
    dist_matrix = euclidean_distance_matrix(coords)
    
    route = list(range(n))
    random.shuffle(route)
    

    best_route = route[:]  # Make a copy
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
    
     
    NUM_CITIES = 30
    GRID_SIZE = 100
    
    np.random.seed(42)  
    coords = np.random.rand(NUM_CITIES, 2) * GRID_SIZE

    print(f"Running Simulated Annealing for {NUM_CITIES} cities...")
    best_route, best_cost, history = simulated_annealing(coords)
    print(f"Best tour length: {best_cost:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("iteration")
    plt.ylabel("Best Tour Length")
    plt.title("Simulated Annealing Progress")
    plt.show()
    
    plt.figure(figsize=(6, 6))
    ordered_coords = coords[best_route + [best_route[0]]]
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-')  
    plt.title(f"Final Tour (Length = {best_cost:.2f})")
    plt.show()