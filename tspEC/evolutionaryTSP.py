import random
import math

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

dataset = []
NODE_NUM = 127
POP_SIZE = NODE_NUM**2

with open("TSPDATA.txt", "r") as file:
    tmp = []
    for line in file:
        tmp = line.strip().split()
        try:
            value1, value2, value3 = map(int, tmp)
            dataset.append(tmp)
        except: pass

def evaluate_permutation(permutation):

    length = 0
    for i in range(0, NODE_NUM):
        c1 = permutation[i]
        c2 = permutation[(i + 1) % NODE_NUM]
        length += math.sqrt((float(dataset[c1][1]) - float(dataset[c2][1]))**2 + (float(dataset[c1][2]) - float(dataset[c2][2]))**2)
    return length,

# weights=(-1.0) means fitness is trying to be minimized
# this block is pretty much boilerplate
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NODE_NUM), NODE_NUM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_permutation)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(POP_SIZE)
num_generations = 5000
for gen in range(num_generations): 
    algorithms.eaSimple(population, toolbox, cxpb=0.2, mutpb=0.3, ngen=1, stats=None, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    print("generation: ", gen)
    print("Fitness value:", best_individual.fitness.values[0])
    if (gen % 100 == 0): 
        print("Best permutation:", best_individual)
print("Fitness value:", best_individual.fitness.values[0])


