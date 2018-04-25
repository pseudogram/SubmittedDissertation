from deap import creator
from deap import base
from deap import tools
import numpy as np
from time import time



# # ------------------------------------------------------------------------------
# #                        SET UP: OpenAI Gym Environment
# # ------------------------------------------------------------------------------
#
# ENV_NAME = 'Pendulum-v0'
# EPISODES = 3  # Number of times to run envionrment when evaluating
# STEPS = 200  # Max number of steps to run run simulation
#
# env = gym.make(ENV_NAME)
#
# # Used to create controller
# obs_dim = env.observation_space.shape[0]  # Input to controller (observ.)
# action_dim = env.action_space.shape[0]  # Output from controller (action)
# nodes = 10  # Unconnected nodes in network in the
#
# # ------------------------------------------------------------------------------
# #                          SET UP: TensorFlow Basic_rnn
# # ------------------------------------------------------------------------------
# # agent = Basic_rnn(obs_dim, action_dim, nodes, dt, False)
# # agent = Linear(obs_dim, action_dim)
# # agent = MLP(obs_dim, action_dim, [3,3,3])
# agent = rnn(obs_dim, action_dim, 10)
#
# # ------------------------------------------------------------------------------
# #                               SET UP GA PARAMETERS
# # ------------------------------------------------------------------------------
# POPULATION_SIZE = 40
# CROSS_PROB = 0.5
# NUM_GEN = 10000   # Number of generations
# DEME_SIZE = 3  # from either side
# MUTATION_RATE = 0.4 # PERCENTAGE OF GENES TO MUTATE
#
# # ----------------------------------------------------------------------------
# #                               CREATE GA
# ------------------------------------------------------------------------------


class Microbial:

    def __init__(self, agent, evaluate, POPULATION_SIZE=40, CROSS_PROB=0.5,
                 NUM_GEN=10000, DEME_SIZE=3, MUTATION_RATE=None):
        """If mutationrate is none, set to 1/Num_Params
        evaluate is the evaluation function"""

        self.NUM_PARAMS = agent.num_params
        self.POPULATION_SIZE = POPULATION_SIZE
        self.CROSS_PROB = CROSS_PROB
        self.NUM_GEN = int(NUM_GEN * POPULATION_SIZE / 2)
        self.DEME_SIZE = DEME_SIZE
        if MUTATION_RATE is None:
            self.MUTATION_RATE = 1/self.NUM_PARAMS

        # Creates a Fitness class, with a weights attribute.
        #    1 means a metric that needs to be maximized = the value
        #    -1 means the metric needs to be minimized. All OpenAI
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Creates a class Individual, that is based on a numpy array as that is the
        # class used for the weights passed to TensorFlow.
        #   It also has an attribute that is a Fitness, when setting the attribute
        #   it automatically calls the __init__() in Fitness initializing the
        #   weight (1)
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        # ==============================================================================

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_floats", lambda n: (np.random.rand(
            n).astype(
            np.float32)-0.5)*2, self.NUM_PARAMS)

        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attr_floats)

        # create a function 'population' that generates a list of individuals, x long.
        #   NB: As repeat init takes 3 parameters, and as the first 2 have been given.
        #       only one is needed.
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)

        self.toolbox.register('evaluate', evaluate, agent=agent)
        self.toolbox.register("crossover", self.cross)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.selDeme, deme_size=self.DEME_SIZE)

    def cross(self, winner, loser):
        """Apply a crossover operation on input sets. The first child is the
        intersection of the two sets, the second child is the difference of the
        two sets.

        Is ind1 the winner or ind2???
        """

        for i in range(self.NUM_PARAMS):
            if np.random.rand() < self.CROSS_PROB:
                loser[i] = winner[i]
        return loser

    def mutate(self, individual):
        """Adds or subtracts 1% with a chance of 1/NUM_PARAMS"""
        # TODO: Increase speed of mutation
        for i in range(self.NUM_PARAMS):
            if np.random.rand() < (self.MUTATION_RATE):
                individual[i] += individual[i] * (np.random.rand()-0.5)*0.01
        return individual

    def selDeme(self, individuals, deme_size):
        """Select *k* individuals at random from the input *individuals* with
        replacement. The list returned contains references to the input
        *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        one = np.random.randint(self.POPULATION_SIZE)
        _next = np.random.randint(1, deme_size + 1)
        if np.random.rand() < 0.5: _next = -_next
        two = (one + _next) % self.POPULATION_SIZE
        return individuals[one], individuals[two]

    def run(self, NUM_GEN):
        if NUM_GEN is None:
            NUM_GEN = self.NUM_GEN
        nevals = int(self.POPULATION_SIZE / 2)
        NUM_GEN *= nevals
        hof = tools.HallOfFame(3, np.array_equal)  # Store the best 3

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        print(stats)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        pop = self.toolbox.population(n=self.POPULATION_SIZE)
        CXPB, MUTPB = 0.5, 0.2

        time_to_compute = time()
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            # Set fitness property of individual to fit.
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=self.POPULATION_SIZE, **record)

        # Start Tournament
        for g in range(1, NUM_GEN+1):
            if g % nevals == 0 :
                print(logbook.stream)

            chosen = self.toolbox.select(pop)
            if chosen[0].fitness > chosen[1].fitness:
                winner = chosen[0]
                loser = chosen[1]
            elif chosen[1].fitness > chosen[0].fitness:
                winner = chosen[1]
                loser = chosen[0]
            else:
                continue
            del loser.fitness.values
            # if g % 100 == 0: print(winner.fitness.values)
            # print(winner.fitness.valid)

            # Apply crossover
            self.toolbox.crossover(winner, loser)

            # Apply mutation
            self.toolbox.mutate(loser)

            loser.fitness.values = self.toolbox.evaluate(loser)

            if g % nevals == 0:
                hof.update(pop)
                record = stats.compile(pop)
                logbook.record(gen=g/nevals, nevals=nevals, **record)

        if g % nevals:
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=(g+1)/nevals, nevals=g, **record)

        time_to_compute -= time()

        print(logbook.stream)
        return pop, logbook, hof, -time_to_compute


