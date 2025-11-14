import random

def fitness(state):
    n = len(state)
    total_pairs = n * (n - 1) // 2
    attacking = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                attacking += 1
                
    return total_pairs - attacking

def random_state(n):
    return [random.randint(0, n - 1) for _ in range(n)]

def simple_hill_climbing(n):
    current_state = random_state(n)
    current_fitness = fitness(current_state)
    
    while True:
        best_neighbor = None
        best_neighbor_fitness = current_fitness

        for col in range(n):
            original_row = current_state[col]
            
            for row in range(n):
                if original_row == row:
                    continue
                    
                neighbor = current_state[:]
                neighbor[col] = row
                score = fitness(neighbor)
            
            
                if score > best_neighbor_fitness:
                    best_neighbor_fitness = score
                    best_neighbor = neighbor

        if best_neighbor is None or best_neighbor_fitness <= current_fitness:
            break
        
        current_state, current_fitness = best_neighbor, best_neighbor_fitness

    return current_state, current_fitness

def print_board(state):
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
    N = 4
    solution, score = simple_hill_climbing(N)

    print(f"Final Solution (row positions): {solution}")
    print(f"Fitness (non-attacking pairs): {score}")
    
    goal_fitness = (N * (N - 1)) // 2
    if score == goal_fitness:
        print("A Valid Solution was Found (this is rare!)")
    else:
        print(f"Stopped at a Local Maximum (Goal is {goal_fitness})")

    print("\nVisual Board:")
    print_board(solution)