import os
import random

from deap import base
from deap import creator
from deap import tools

from matsuoka_walk import oscillator_nw, Logger, log

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the logging variables
# This also creates a new log file
Logger(log_dir=os.path.join(home_dir, '.bio_walk/logs/'), log_flag=True)

# Create the position bounds of the individual
log('[GA] Creating position bounds')
FLT_MIN_KF,    FLT_MAX_KF    = 0.2, 0.5
FLT_MIN_GAIN1, FLT_MAX_GAIN1 = 0.01, 1.0
FLT_MIN_GAIN2, FLT_MAX_GAIN2 = 0.01, 1.0
FLT_MIN_GAIN3, FLT_MAX_GAIN3 = 0.01, 1.0
FLT_MIN_GAIN4, FLT_MAX_GAIN4 = 0.01, 1.0
FLT_MIN_GAIN5, FLT_MAX_GAIN5 = 0.01, 1.0
FLT_MIN_GAIN6, FLT_MAX_GAIN6 = 0.01, 1.0
FLT_MIN_BIAS1, FLT_MAX_BIAS1 = -0.6, 0.0
FLT_MIN_BIAS2, FLT_MAX_BIAS2 = 0.0, 0.5
FLT_MIN_BIAS3, FLT_MAX_BIAS3 = -0.5, 0.0
FLT_MIN_BIAS4, FLT_MAX_BIAS4 = 0.0, 1.0

log('[GA] Logging position bounds')
log('[GA] FLT_MIN_KF={0}, FLT_MAX_KF={1}'.format(FLT_MIN_KF, FLT_MAX_KF))
log('[GA] FLT_MIN_GAIN1={0}, FLT_MAX_GAIN1={1}'.format(FLT_MIN_GAIN1, FLT_MAX_GAIN1))
log('[GA] FLT_MIN_GAIN2={0}, FLT_MAX_GAIN2={1}'.format(FLT_MIN_GAIN2, FLT_MAX_GAIN2))
log('[GA] FLT_MIN_GAIN3={0}, FLT_MAX_GAIN3={1}'.format(FLT_MIN_GAIN3, FLT_MAX_GAIN3))
log('[GA] FLT_MIN_GAIN4={0}, FLT_MAX_GAIN4={1}'.format(FLT_MIN_GAIN4, FLT_MAX_GAIN4))
log('[GA] FLT_MIN_GAIN5={0}, FLT_MAX_GAIN5={1}'.format(FLT_MIN_GAIN5, FLT_MAX_GAIN5))
log('[GA] FLT_MIN_GAIN6={0}, FLT_MAX_GAIN6={1}'.format(FLT_MIN_GAIN6, FLT_MAX_GAIN6))
log('[GA] FLT_MIN_BIAS1={0}, FLT_MAX_BIAS1={1}'.format(FLT_MIN_BIAS1, FLT_MAX_BIAS1))
log('[GA] FLT_MIN_BIAS2={0}, FLT_MAX_BIAS2={1}'.format(FLT_MIN_BIAS2, FLT_MAX_BIAS2))
log('[GA] FLT_MIN_BIAS3={0}, FLT_MAX_BIAS3={1}'.format(FLT_MIN_BIAS3, FLT_MAX_BIAS3))
log('[GA] FLT_MIN_BIAS4={0}, FLT_MAX_BIAS4={1}'.format(FLT_MIN_BIAS4, FLT_MAX_BIAS4))

# Define a custom class named `FitnessMax`
# Single objective function is specified by the tuple `weights=(1.0,)`
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Create a class named `Individual` which inherits from the class `list` and has `FitnessMax` as an attribute
creator.create("Individual", list, fitness=creator.FitnessMax)

# Now we will use our custom classes to create types representing our individuals as well as our whole population.
# All the objects we will use on our way, an individual, the population, as well as all functions, operators, and
# arguments will be stored in a DEAP container called `Toolbox`. It contains two methods for adding and removing
# content, register() and unregister().

toolbox = base.Toolbox()

# Attribute generator - specify how each single gene is to be created
toolbox.register("kf_flt", random.uniform, FLT_MIN_KF, FLT_MAX_KF)
toolbox.register("gain1_flt", random.uniform, FLT_MIN_GAIN1, FLT_MAX_GAIN1)
toolbox.register("gain2_flt", random.uniform, FLT_MIN_GAIN2, FLT_MAX_GAIN2)
toolbox.register("gain3_flt", random.uniform, FLT_MIN_GAIN3, FLT_MAX_GAIN3)
toolbox.register("gain4_flt", random.uniform, FLT_MIN_GAIN4, FLT_MAX_GAIN4)
toolbox.register("gain5_flt", random.uniform, FLT_MIN_GAIN5, FLT_MAX_GAIN5)
toolbox.register("gain6_flt", random.uniform, FLT_MIN_GAIN6, FLT_MAX_GAIN6)
toolbox.register("bias1_flt", random.uniform, FLT_MIN_BIAS1, FLT_MAX_BIAS1)
toolbox.register("bias2_flt", random.uniform, FLT_MIN_BIAS2, FLT_MAX_BIAS2)
toolbox.register("bias3_flt", random.uniform, FLT_MIN_BIAS3, FLT_MAX_BIAS3)
toolbox.register("bias4_flt", random.uniform, FLT_MIN_BIAS4, FLT_MAX_BIAS4)

# Specify the structure of an individual chromosome
N_CYCLES=1 # Number of times to repeat this pattern

# Specify the sequence of genes in an individual chromosome
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.kf_flt,
                  toolbox.gain1_flt, toolbox.gain2_flt, toolbox.gain3_flt,
                  toolbox.gain4_flt, toolbox.gain5_flt, toolbox.gain6_flt,
                  toolbox.bias1_flt ,toolbox.bias2_flt, toolbox.bias3_flt, toolbox.bias4_flt),
                 n=N_CYCLES)

# Define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the goal / fitness function
toolbox.register("evaluate", oscillator_nw)

# Register the crossover operator - 2 point crossover is used here
toolbox.register("mate", tools.cxTwoPoint)

# Register a mutation operator
# Mutation is done by adding a float to each gene. This float to be added is randomly selected from a Gaussian
# distribution with mu=0.0 and sigma=0.01
# Probability of mutation is 0.05
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.01, indpb=0.05)

# Operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of 3 individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# Size of the population
POP_SIZE = 200

# Maximum generations
MAX_GEN = 100

def main():
    #random.seed(64)

    # Create an initial population of `POP_SIZE` individuals (where each individual is a list of floats)
    pop = toolbox.population(n=POP_SIZE)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.1

    log('[GA] Starting genetic algorithm')

    # Evaluate the entire population and store the fitness of each individual
    log('[GA] Finding the fitness of individuals in the initial generation')
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        print ind, fit
        ind.fitness.values = (fit,)

    # Extracting all the fitnesses
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    best_ind_ever = None
    best_fitness_ever = 0.0

    # Begin the evolution
    while max(fits) < 100 and g < MAX_GEN:

        # A new generation
        g = g + 1
        log('[GA] Running generation {0}'.format(g))

        # Select the next generation individuals
        log('[GA] Selecting the next generation')
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

        # Since the content of some of our offspring changed during the last step, we now need to
        # re-evaluate their fitnesses. To save time and resources, we just map those offspring which
        # fitnesses were marked invalid.
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        log('[GA] Evaluated {0} individuals (invalid fitness)'.format(len(invalid_ind)))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        log('[GA] Results for generation {0}'.format(g))
        log('[GA] Min %s' % min(fits))
        log('[GA] Max %s' % max(fits))
        log('[GA] Avg %s' % mean)
        log('[GA] Std %s' % std)

        best_ind_g = tools.selBest(pop, 1)[0]

        # Store the best individual over all generations
        if best_ind_g.fitness.values[0] > best_fitness_ever:
            best_fitness_ever = best_ind_g.fitness.values[0]
            best_ind_ever = best_ind_g

        log('[GA] Best individual for generation {0}: {1}, {2}'.format(g, best_ind_g, best_ind_g.fitness.values[0]))

        log('[GA] ############################# End of generation {0} #############################'.format(g))

    log('[GA] ===================== End of evolution =====================')

    best_ind = tools.selBest(pop, 1)[0]
    log('[GA] Best individual in the population: %s, %s' % (best_ind, best_ind.fitness.values[0]))
    log('[GA] Best individual ever: %s, %s' % (best_ind_ever, best_fitness_ever))

if __name__ == "__main__":
    main()
