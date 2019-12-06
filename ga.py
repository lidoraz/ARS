import numpy as np
import uuid
from itertools import combinations

"""
Generic FakeUserGeneticAlgorithm class

supports init , selection, crossover mutation
need only to create a fitness function, and assign a fitness value for each agent

"""



class AttackAgent:
    def __init__(self, n_m_users=0, n_items=0, gnome=None, d_birth=0, POS_RATIO=1, BINARY = True):
        if gnome is not None and ( n_items != 0 or n_items != 0 or d_birth == 0 or POS_RATIO != 1):
            raise ValueError('not valid config')
        self.fitness = -42  # fitness is always between -1 to 1.
        # self.is_fame = False
        self.evaluted = False
        self.d_birth = d_birth
        # changing params through generations
        self.age = 0  # at what generation this agent created?
        self.generations_mutated = 0
        self.id = uuid.uuid4().hex[:8]
        # initiate or create one from given gnome (offspring)
        if gnome is None:
            if BINARY:
                self.gnome = np.random.choice([0, 1],
                                              size=(n_m_users, n_items),
                                              p=[1 - POS_RATIO, POS_RATIO])
            else:
                self.gnome = np.random.choice(list(range(6)),
                                              size=(n_m_users, n_items),
                                              p=[1-POS_RATIO, POS_RATIO / 5, POS_RATIO / 5, POS_RATIO / 5, POS_RATIO / 5, POS_RATIO / 5])
        else:
            self.gnome = gnome

class FakeUserGeneticAlgorithm:
    def __init__(self, POP_SIZE,MAX_POP_SIZE, N_GENERATIONS, SELECTION_GENERATIONS_BEFORE_REMOVAL,
                 SELECTION_REMOVE_PERCENTILE, MUTATE_USER_PROB, MUTATE_BIT_PROB, CONVERT_BINARY, POS_RATIO,CROSSOVER_CREATE_TOP):

        self.POP_SIZE = POP_SIZE
        self.MAX_POP_SIZE = MAX_POP_SIZE
        self.N_GENERATIONS = N_GENERATIONS
        self.SELECTION_GENERATIONS_BEFORE_REMOVAL = SELECTION_GENERATIONS_BEFORE_REMOVAL
        self.SELECTION_REMOVE_PERCENTILE = SELECTION_REMOVE_PERCENTILE
        self.MUTATE_USER_PROB = MUTATE_USER_PROB
        self.MUTATE_BIT_PROB = MUTATE_BIT_PROB
        self.CONVERT_BINARY = CONVERT_BINARY
        self.POS_RATIO = POS_RATIO
        self.CROSSOVER_CREATE_TOP = CROSSOVER_CREATE_TOP

        print("Created 'FakeUserGeneticAlgorithm with 'Elitism' - best individual will not be mutated")



    def init_agents(self, n_fake_users, n_items):
        return [AttackAgent(n_fake_users, n_items, POS_RATIO=self.POS_RATIO, BINARY=self.CONVERT_BINARY) for _ in range(self.POP_SIZE)]


    def fitness(self, agents):

        for agent in agents:
            if not agent.evaluted:
                agent.fitness = sum(sum(agent.gnome))
        # train model with this new malicious data
        # eval model
        # continue training until there is no improvment
        # take best model, calulate difference in best model and pert_best model
        # return it as fitness
        return agents


    def selection(self, agents):
        """
            Sorts the pool, removes old indviduals that are over GENERATIONS_BEFORE_REMOVAL and are worse than REMOVE_PERCENTILE in score
            """
        # update age
        for agent in agents:
            agent.age += 1
        # sort by fitness best to worse
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        # get 5% worst
        fitness_treshold = agents[int((1-self.SELECTION_REMOVE_PERCENTILE) * len(agents))].fitness
        # remove agents that are over age and their fitness is low
        #TODO: check this, not should be easier to understand
        remove_func = lambda x: not(x.age > self.SELECTION_GENERATIONS_BEFORE_REMOVAL or x.fitness < fitness_treshold)
        agents_removed_worst = list(filter(remove_func, agents))
        return agents_removed_worst

        # sort by fitness, take 2 best agents
        # keep best one in pool

    @staticmethod
    def pair_crossover(agent1, agent2, cur_generation):
        """
        Make 2 offsprings from 2 best agents
        the cross over will take bunch of users from each attack, and mix them up together,
        the cross over will not change the ratings themselvs, only the rows.
        make also with the hall of fame
        :param agent1:
        :param agent2:
        :param cur_generation:
        :return:
        """
        agent_1_part_prefix = agent1.gnome[:agent1.gnome.shape[0] // 2]
        agent_1_part_postfix = agent1.gnome[agent1.gnome.shape[0] // 2:]
        agent_2_part_prefix = agent2.gnome[:agent2.gnome.shape[0] // 2]
        agent_2_part_postfix = agent2.gnome[agent2.gnome.shape[0] // 2:]
        offspring_1 = AttackAgent(gnome=np.concatenate([agent_1_part_prefix, agent_2_part_postfix]), d_birth= cur_generation)
        offspring_2 = AttackAgent(gnome=np.concatenate([agent_2_part_prefix, agent_1_part_postfix]), d_birth= cur_generation)
        return offspring_1, offspring_2

    def crossover(self, agents, cur_generation):
        new_agents = []
        top_candidates = agents[:self.CROSSOVER_CREATE_TOP]
        for pair in combinations(top_candidates, 2):
            offspring_1, offspring_2 = self.pair_crossover(pair[0], pair[1], cur_generation)
            new_agents.append(offspring_1)
            new_agents.append(offspring_2)
        agents = new_agents + agents
        if self.MAX_POP_SIZE:
            agents = agents[:self.MAX_POP_SIZE]

        return agents, len(new_agents)
    # return offspring_1, offspring_2
    def mutation(self, agents):
        # mutation utility functions
        def bit_flip_func_binary(x):
            # bits = [1, 0]
            if np.random.rand() < self.MUTATE_BIT_PROB:
                if x == 0:
                    return 1
                else:
                    return 0
            else:
                return x

        def bit_flip_func_non_binary(x):
            if np.random.rand() < self.MUTATE_BIT_PROB:
                return np.random.randint(1, 6)
            else:
                return x

        def flip_bit_1d_array(arr):
            if self.CONVERT_BINARY:
                return list(map(bit_flip_func_binary, arr))
            else:
                return list(map(bit_flip_func_non_binary, arr))

        # sort by fitness best to worse - sort again because we added new agents to start
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        for agent in agents[1:]:
            if np.random.rand() < self.MUTATE_USER_PROB:
                agent.gnome = np.apply_along_axis(func1d=flip_bit_1d_array, axis=0, arr=agent.gnome)
                agent.generations_mutated += 1
                agent.evaluted = False
        # flip bit in an entry in a prob
        # this will work on every entry, to create stohastic behaviour, kind of epsilon greedy method.
        return agents

    @staticmethod
    def get_stats_writer(agents, cur_generation, tb):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness for ind in agents]

        length = len(agents)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        max_fit = max(fits)
        min_fit = min(fits)
        tb.add_scalar('max_fit', max_fit, cur_generation)
        tb.add_scalar('min_fit', min_fit, cur_generation)
        tb.add_scalar('mean_fit', mean, cur_generation)
        tb.add_scalar('std_fit', std, cur_generation)
        tb.add_scalar('pool_size', length, cur_generation)
        tb.close()
        return length, min_fit, max_fit, mean, std
        # print(f"G:{cur_generation}\tp_size:{length}\tmin:{min(fits):.2f}\tmax:{max(fits):.2f}\tavg:{mean:.2f}\tstd:{std:.2f}")
        # print(f"Best agent index: {np.argmax(fits)}")

    def print_stats(self, agents, n_created, cur_generation):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness for ind in agents]

        length = len(agents)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        max_fit = max(fits)
        min_fit = min(fits)
        print(f'G={cur_generation}, P_SIZE={length}, n_created={n_created}, min_fit={min_fit:0.3f}, max_fit={max_fit:0.3f}, mean={mean:0.3f}, std={std:0.3f}')
        return length, min_fit, max_fit, mean, std
        # return length, min_fit, max_fit, mean, std
    @staticmethod
    def save(agents, n_fake_users, cur_generation, train_frac, save_dir='agents'):
        import pickle
        import os
        with open(os.path.join(save_dir, f'g_{cur_generation}_n_fake{n_fake_users}_t_{train_frac}_agents{len(agents)}_dump.dmp')) as file:
            pickle.dump(agents, file, 'wb')



def main():
    from tensorboardX import SummaryWriter

    # HYPER-PARAMETERS
    POP_SIZE = 100
    MAX_POP_SIZE = 300
    N_GENERATIONS = 1000
    # Mutation
    MUTATE_USER_PROB = 0.2  # prob for choosing an individual
    MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
    # Selection
    SELECTION_GENERATIONS_BEFORE_REMOVAL = 20
    SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5%
    # Crossover
    CROSSOVER_CREATE_TOP = 5

    #Model / Dataset related
    N_FAKE_USERS = 10
    N_ITEMS = 2000
    # BINARY = False  # binary or non binary data
    POS_RATIO = 0.5 # Ratio pos/ neg ratio  one percent from each user

    # print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    N_GENERATIONS = 1000
    ga = FakeUserGeneticAlgorithm(POP_SIZE=POP_SIZE,
                                  MAX_POP_SIZE=MAX_POP_SIZE,
                                  N_GENERATIONS=N_GENERATIONS,
                                  SELECTION_GENERATIONS_BEFORE_REMOVAL=SELECTION_GENERATIONS_BEFORE_REMOVAL,
                                  SELECTION_REMOVE_PERCENTILE=SELECTION_REMOVE_PERCENTILE,
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=True,
                                  POS_RATIO=POS_RATIO,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP)

    agents = ga.init_agents(n_fake_users=N_FAKE_USERS, n_items= N_ITEMS)
    # tb = SummaryWriter(comment = f'test_{POS_RATIO}_{SELECTION_GENERATIONS_BEFORE_REMOVAL}_{SELECTION_REMOVE_PERCENTILE}_{MUTATE_USER_PROB}_{MUTATE_BIT_PROB}_{CROSSOVER_CREATE_TOP}')
    print('created n_agents', len(agents))
    n_created = 0
    for cur_generation in range(1, N_GENERATIONS):
        agents = ga.fitness(agents)
        # if cur_generation % 3 == 0:
        # ga.get_stats_writer(agents, cur_generation, tb)
        ga.print_stats(agents, n_created, cur_generation)

        agents = ga.selection(agents)
        agents, n_created = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)







# pop = [agent for agent in ]

if __name__ == '__main__':
    main()