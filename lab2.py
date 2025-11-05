import random

def fitness(state):
    """Calculate number of non-attacking pairs of queens."""
    n = len(state)
    total_pairs = n * (n - 1) // 2
    attacking = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check for horizontal or diagonal attack
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                attacking += 1
                
    return total_pairs - attacking
 
    
def random_state(n):
    """Generate a random board configuration."""
    return [random.randint(0, n - 1) for _ in range(n)]

def get_neighbors(state):
    """Generate all possible neighbors by moving one queen in its column."""
    neighbors = []
    n = len(state)
    
    for col in range(n):
        for row in range(n):
            if state[col] != row:
                neighbor = list(state)
                neighbor[col] = row
                neighbors.append(tuple(neighbor)) # Use tuple to be hashable if needed, list is fine too
                
    return neighbors


def hill_climbing(n):
    """Performs one run of steepest-ascent hill climbing."""
    current = random_state(n)
    current_fitness = fitness(current)

    while True:
        neighbors = get_neighbors(current)
        if not neighbors:
            break

        # Create a list of (fitness, neighbor) tuples
        neighbor_data = []
        for state in neighbors:
            neighbor_data.append((fitness(state), state))
        
        if not neighbor_data:
            break

        # Find the neighbor with the best (max) fitness
        best_neighbor_fitness, best_neighbor = max(neighbor_data, key=lambda item: item[0])

        if best_neighbor_fitness <= current_fitness:
            break  # Local maxima reached
        
        # Move to the best neighbor
        current, current_fitness = best_neighbor, best_neighbor_fitness

    return current, current_fitness

def random_restart_hill_climbing(n, max_restarts=1000):
    """Runs hill climbing multiple times to find a global optimum."""
    best_solution = None
    best_fitness = -1
    goal_fitness = (n * (n - 1)) // 2 # Max non-attacking pairs

    for restart in range(max_restarts):
        solution, score = hill_climbing(n)
        
        if score > best_fitness:
            best_solution, best_fitness = solution, score

        if best_fitness == goal_fitness:
            print(f"Solution found after {restart+1} restarts")
            return best_solution, best_fitness

    print("Max restarts reached, best solution found so far")
    return best_solution, best_fitness
 

def print_board(state):
    """Print the chessboard with queens."""
    n = len(state)
    for row in range(n):
        line = ""
        for col in range(n):
            if state[col] == row:
                line += " Q "
            else:
                line += " . "
        print(line)
    print("\n")


if __name__ == "__main__":
    N = 8  # Set board size
    
    solution, score = random_restart_hill_climbing(N, max_restarts=500)

    print(f"Final Solution (row positions of queens): {list(solution)}")
    print(f"Fitness (non-attacking pairs): {score}")
    
    if score == (N * (N - 1)) // 2:
        print("Valid Solution Found")
    else:
        print("Local Maxima but Best Found")

    print("\nVisual Board:")
    print_board(solution)
