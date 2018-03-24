import random

from deap import base
from deap import creator
from deap import tools

from matsuoka_walk.fitness import hart6sc

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

FLT_MIN_1, FLT_MAX_1 = 0.0, 1.0
FLT_MIN_2, FLT_MAX_2 = 0.0, 1.0
FLT_MIN_3, FLT_MAX_3 = 0.0, 1.0
FLT_MIN_4, FLT_MAX_4 = 0.0, 1.0
FLT_MIN_5, FLT_MAX_5 = 0.0, 1.0
FLT_MIN_6, FLT_MAX_6 = 0.0, 1.0

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
#toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("attr_flt_1", random.uniform, FLT_MIN_1, FLT_MAX_1)
toolbox.register("attr_flt_2", random.uniform, FLT_MIN_2, FLT_MAX_2)
toolbox.register("attr_flt_3", random.uniform, FLT_MIN_3, FLT_MAX_3)
toolbox.register("attr_flt_4", random.uniform, FLT_MIN_4, FLT_MAX_4)
toolbox.register("attr_flt_5", random.uniform, FLT_MIN_5, FLT_MAX_5)
toolbox.register("attr_flt_6", random.uniform, FLT_MIN_6, FLT_MAX_6)

N_CYCLES=1

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_flt_1, toolbox.attr_flt_2, toolbox.attr_flt_3,
                  toolbox.attr_flt_4, toolbox.attr_flt_5, toolbox.attr_flt_6),
                 n=N_CYCLES)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)

# define the population to be a list of individuals
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return sum(individual),


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", hart6sc)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.01, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


# ----------

def main():
    #random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.1

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < 200:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()
