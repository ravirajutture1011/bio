import random
import math

def objective_function(x):
    return x * math.sin(10 * math.pi * x) + 2.0

def initialize_population(pop_size, bounds):
    return [random.uniform(bounds[0], bounds[1]) for _ in range(pop_size)]

def fitness(x, maximize=True):
    val = objective_function(x)
    return val if maximize else -val

def selection(pop, maximize=True):
    i, j = random.sample(range(len(pop)), 2)
    if fitness(pop[i], maximize) > fitness(pop[j], maximize):
        return pop[i]
    else:
        return pop[j]

def crossover(parent1, parent2, crossover_rate=0.9):
    if random.random() < crossover_rate:
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    else:
        return parent1

def mutate(x, mutation_rate=0.1, bounds=(-1, 2)):
    if random.random() < mutation_rate:
        x = x + random.uniform(-0.1, 0.1)
        x = max(bounds[0], min(bounds[1], x))
    return x

def genetic_algorithm(pop_size=50, generations=100, bounds=(-1, 2), maximize=True):
    population = initialize_population(pop_size, bounds)
    
    best = population[0]
    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1 = selection(population, maximize)
            parent2 = selection(population, maximize)
            child = crossover(parent1, parent2)
            child = mutate(child, bounds=bounds)
            new_population.append(child)
        population = new_population
        
        for ind in population:
            if fitness(ind, maximize) > fitness(best, maximize):
                best = ind
        
        if gen % 10 == 0:
            print(f"Generation {gen}, Best: x={best:.5f}, f(x)={objective_function(best):.5f}")
            
    return best

best_solution = genetic_algorithm()
print(f"\nOptimal Solution Found:")
print(f"x = {best_solution:.5f}, f(x) = {objective_function(best_solution):.5f}")

