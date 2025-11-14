
import numpy as np
import matplotlib.pyplot as plt

 
def fitness_function(pop, adjacency_matrix):
    conflicts = 0
    for i in range(len(pop)):
        for j in range(len(pop)):
            if adjacency_matrix[i][j] == 1 and pop[i] == pop[j]:
                conflicts += 1
    return 1 / (1 + conflicts)   

 
def selection(population, fitness_values):
    probs = fitness_values / np.sum(fitness_values)
    idx = np.random.choice(len(population), size=2, p=probs, replace=False)
    return population[idx[0]], population[idx[1]]

 
def crossover(parent1, parent2):
    mask = np.random.randint(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

 
def mutate(child, num_colors, mutation_rate=0.1):
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.randint(0, num_colors)
    return child
 
def genetic_graph_coloring(adjacency_matrix, num_colors=4, pop_size=90, generations=700):
    num_nodes = len(adjacency_matrix)
    population = [np.random.randint(0, num_colors, size=num_nodes) for _ in range(pop_size)] # (start,end,no of times)
    best_fitness_history = []

    for gen in range(generations):
        fitness_values = np.array([fitness_function(pop, adjacency_matrix) for pop in population])
        best_fitness = np.max(fitness_values)
        best_fitness_history.append(best_fitness)

        new_population = []
        for _ in range(pop_size // 2):
            p1, p2 = selection(population, fitness_values)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, num_colors))
            new_population.append(mutate(c2, num_colors))
        population = new_population

    best_idx = np.argmax(fitness_values)
    return population[best_idx], best_fitness, best_fitness_history

 
adjacency_matrix = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

best_solution, best_fit, history = genetic_graph_coloring(adjacency_matrix, num_colors=3, generations=200)

print("Best Coloring:", best_solution)
print("Best Fitness:", best_fit)

 
plt.plot(history)
plt.title("Convergence Curve (Graph Coloring using GA)")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()