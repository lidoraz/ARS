import numpy as np
import uuid
from itertools import combinations
import logging
from time import time
logger = logging.getLogger('ga_attack')

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
        self.selected = False
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
    def __init__(self, ga_params, tb, baseline):
        self.baseline_fit = baseline
        self.tb = tb
        self.POP_SIZE =                             ga_params['POP_SIZE']
        self.N_GENERATIONS =                        ga_params['N_GENERATIONS']
        self.SELECTION_MODE =                       ga_params['SELECTION_MODE']
        self.SELECTION_GENERATIONS_BEFORE_REMOVAL = ga_params['SELECTION_GENERATIONS_BEFORE_REMOVAL']
        self.SELECTION_REMOVE_PERCENTILE =          ga_params['SELECTION_REMOVE_PERCENTILE']
        self.MUTATE_USER_PROB =                     ga_params['MUTATE_USER_PROB']
        self.MUTATE_BIT_PROB =                      ga_params['MUTATE_BIT_PROB']
        self.CONVERT_BINARY =                       ga_params['CONVERT_BINARY']
        self.POS_RATIO =                            ga_params['POS_RATIO']
        self.MAX_POS_RATIO = ga_params['MAX_POS_RATIO']
        self.CROSSOVER_CREATE_TOP =                 ga_params['CROSSOVER_CREATE_TOP']

        self.__fitness_norm_list = None
        self.best_max_fit = 0
        self.best_max_fit_g = 0
        self.best_max_mean_rating_ratio = 0
        self.n_keep_best = 3
        self.crossover_type = ga_params['crossover_type']
        self.start_time = time()
        self.curr_generation_time = self.start_time

        if self.SELECTION_MODE == 'TOURNAMENT':
            self.n_tournement_creates = self.CROSSOVER_CREATE_TOP * (self.CROSSOVER_CREATE_TOP - 1)
            logger.info(f"Selection='TOURNAMENT'.. Will create {self.n_tournement_creates} new agents every generation, remove them according to parameters")
            logger.info(f'Maximum pool size: {self.POP_SIZE} (if 0 - do not kill bad performing individuals because space limit.')
            logger.info("Created 'FakeUserGeneticAlgorithm with 'Elitism' - best individual will not be mutated")
        elif self.SELECTION_MODE == 'ROULETTE':
            logger.info(f"Selection='ROULETTE'.. Will re-create the population every generation based on normalized probabilities")
        elif self.SELECTION_MODE == 'RANDOM':
            logger.info("*******Selection='RANDOM'.. simulates a complete random behaviour, GA will not utilize")
        else:
            raise ValueError(f'SELECTION_MODE={self.SELECTION_MODE} not supported')



    def init_agents(self, n_fake_users, n_items):
        self.N_ITEMS = n_items
        self.MAX_ALLOWED_INTERCATIONS = self.N_ITEMS * self.MAX_POS_RATIO
        return [AttackAgent(n_fake_users, n_items, POS_RATIO=self.POS_RATIO, BINARY=self.CONVERT_BINARY) for _ in range(self.POP_SIZE)]

    def fitness(self, agents):
        for agent in agents:
            if not agent.evaluted:
                agent.fitness = sum(sum(agent.gnome))
                agent.evaluted = True
        return agents

    def selection_roulette(self, agents):
        sum_fitness = sum(list(map(lambda x: x.fitness, agents)))  # work around for negative probabilities
        if sum_fitness != 0:
            for agent in agents:
                assert agent.fitness >= 0, "Fitness cannot be negative"
                agent.fitness_norm = (agent.fitness) / (sum_fitness)
            self.__fitness_norm_list = np.array(list(map(lambda x: x.fitness_norm, agents)))
        else:
            self.__fitness_norm_list = np.full(len(agents), 1/len(agents))
        return agents

    def selection_tournament(self, agents):
        """
        Sorts the pool, removes old indviduals that are over GENERATIONS_BEFORE_REMOVAL and are worse than REMOVE_PERCENTILE in score
        """
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)  # sort by fitness best to worse
        for agent in agents[:self.CROSSOVER_CREATE_TOP]:
            agent.selected = True
        return agents

    def selection(self, agents):
        if self.SELECTION_MODE == 'TOURNAMENT':
            return self.selection_tournament(agents)
        elif self.SELECTION_MODE == 'ROULETTE':
            return self.selection_roulette(agents)
        else:
            raise ValueError('selection - not supported mode')

    def pair_crossover_simple(self, agent1, agent2, cur_generation):
        """
        Make 2 offsprings from 2 best agents
        the cross over will take bunch of users from each attack, and mix them up together,
        the cross over will not change the ratings themselvs, only the rows.
        make also with the hall of fame
        """
        n_users = agent1.gnome.shape[0]
        agent_1_part_prefix = agent1.gnome[:n_users // 2, :]
        agent_1_part_postfix = agent1.gnome[n_users // 2:, :]
        agent_2_part_prefix = agent2.gnome[:n_users // 2, :]
        agent_2_part_postfix = agent2.gnome[n_users // 2:, :]
        offspring_1 = AttackAgent(gnome=np.concatenate([agent_1_part_prefix, agent_2_part_postfix]), d_birth= cur_generation)
        offspring_2 = AttackAgent(gnome=np.concatenate([agent_2_part_prefix, agent_1_part_postfix]), d_birth= cur_generation)
        return offspring_1, offspring_2

    def choose_interactions(self, row):
        locations = np.nonzero(row)[0]
        indicies_to_be_sampled = int(len(locations) - self.MAX_ALLOWED_INTERCATIONS)
        if indicies_to_be_sampled > 0:
            # print(indicies_to_be_sampled)
            places = np.random.choice(locations, indicies_to_be_sampled, replace=False)
            np.put(row, places, 0)
        return row

    def pair_crossover_items(self, agent1, agent2, cur_generation):
        n_items = agent1.gnome.shape[1]

        agent_1_part_prefix = agent1.gnome[:, n_items // 2:]
        agent_1_part_postfix = agent1.gnome[:, :n_items // 2]
        agent_2_part_prefix = agent2.gnome[:, n_items // 2:]
        agent_2_part_postfix = agent2.gnome[:, :n_items // 2]
        # make sure it is valid, otherwise, sample ratings:
        gnome1 = np.concatenate([agent_1_part_prefix, agent_2_part_postfix], axis=1)
        gnome2 = np.concatenate([agent_2_part_prefix, agent_1_part_postfix], axis=1)
        gnome1 = np.apply_along_axis(func1d=lambda row: self.choose_interactions(row), axis=1,
                                     arr=gnome1)
        gnome2 = np.apply_along_axis(func1d=lambda row: self.choose_interactions(row), axis=1,
                                     arr=gnome2)
        offspring_1 = AttackAgent(gnome=gnome1,
                                  d_birth=cur_generation)
        offspring_2 = AttackAgent(gnome=gnome2,
                                  d_birth=cur_generation)
        return offspring_1, offspring_2

    @staticmethod
    def pair_crossover_bit(agent1, agent2, cur_generation):
        agent_1_part_prefix = agent1.gnome[:agent1.gnome.shape[0] // 2]
        agent_1_part_postfix = agent1.gnome[agent1.gnome.shape[0] // 2:]
        agent_2_part_prefix = agent2.gnome[:agent2.gnome.shape[0] // 2]
        agent_2_part_postfix = agent2.gnome[agent2.gnome.shape[0] // 2:]
        offspring_1 = AttackAgent(gnome=np.concatenate([agent_1_part_prefix, agent_2_part_postfix]),
                                  d_birth=cur_generation)
        offspring_2 = AttackAgent(gnome=np.concatenate([agent_2_part_prefix, agent_1_part_postfix]),
                                  d_birth=cur_generation)
        return offspring_1, offspring_2

    def pair_crossover(self, agent1, agent2, cur_generation):
        if self.crossover_type == 'simple':
            offspring_1, offspring_2 = self.pair_crossover_simple(agent1, agent2, cur_generation)
        elif self.crossover_type == 'items':
            offspring_1, offspring_2 = self.pair_crossover_items(agent1, agent2, cur_generation)
        else:
            raise ValueError('not valid crossover_type:', self.crossover_type)
        return offspring_1, offspring_2


    def crossover(self, agents, cur_generation):
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        new_agents = []
        if self.SELECTION_MODE == 'TOURNAMENT':
            top_candidates = [agent for agent in agents if agent.selected]
            for agent in top_candidates:
                agent.selected = False
            for pair in combinations(top_candidates, 2):
                offspring_1, offspring_2 = self.pair_crossover(pair[0], pair[1], cur_generation)
                new_agents.append(offspring_1)
                new_agents.append(offspring_2)
            new_agents = new_agents + agents
            if self.POP_SIZE:
                new_agents = new_agents[:len(new_agents) - self.n_tournement_creates]

        elif self.SELECTION_MODE == 'ROULETTE':
            for i in range(self.n_keep_best):
                new_agents.append(agents[i])
            # sample MAX_POOL SIZE agents from distribution of fitness_norm.
            while len(new_agents) < self.POP_SIZE:
                idx1, idx2 = np.random.choice(np.arange(len(agents)), p=self.__fitness_norm_list, size=(2,))
                if idx1 != idx2:
                    offspring_1, offspring_2 = self.pair_crossover(agents[idx1], agents[idx2], cur_generation)
                    new_agents.append(offspring_1)
                    new_agents.append(offspring_2)

        for agent in new_agents:
            agent.age += 1
        return new_agents

    def mutation(self, agents):
        """
        Mutation that has a hard regulaizer on total amount of allowed interactions on every user
        :param agents:
        :return:
        """
        def bit_flip_func_non_binary(x):
            if np.random.rand() < self.MUTATE_BIT_PROB:
                return np.random.randint(1, 6)
            else:
                return x

        def bit_flip_func_binary(arr):
            arr_copy = np.copy(arr)
            total_interations = np.count_nonzero(arr_copy)
            for i in range(len(arr_copy)):
                flip = np.random.rand() < self.MUTATE_BIT_PROB
                if flip:
                    if arr_copy[i] == 0:
                        if total_interations < self.MAX_ALLOWED_INTERCATIONS:
                            total_interations += 1
                            arr_copy[i] = 1
                    else:
                        total_interations -= 1
                        arr_copy[i] = 0

            return arr_copy

        def flip_bit_1d_array(arr):
            if self.CONVERT_BINARY:
                return bit_flip_func_binary(arr)
            else:
                return list(map(bit_flip_func_non_binary, arr))

        # sort by fitness best to worse - sort again because we added new agents to start
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        for agent in agents[1:]:
            if np.random.rand() < self.MUTATE_USER_PROB:
                agent.gnome = np.apply_along_axis(func1d=flip_bit_1d_array, axis=1, arr=agent.gnome)
                agent.generations_mutated += 1
                agent.evaluted = False
        # flip bit in an entry in a prob
        # this will work on every entry, to create stohastic behaviour, kind of epsilon greedy method.
        return agents

    def get_stats_writer(self, agents, cur_generation):
        """
        watch correlation between increase in fit, to an increase in delta hr
        :param agents:
        :param cur_generation:
        :return:
        """
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness for ind in agents]
        hrs = [ind.fitness for ind in agents]
        length = len(agents)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        max_fit = max(fits)
        min_fit = min(fits)
        best_agent_id = np.argmax(fits)
        max_dhr = agents[best_agent_id].delta_hr
        max_ndcg = agents[best_agent_id].delta_ndcg
        found_new_best = False
        if max_fit > self.best_max_fit:
            self.best_max_fit = max_fit
            self.best_max_fit_g = cur_generation
            self.best_max_hr = agents[best_agent_id].delta_hr
            self.best_max_ndcg = agents[best_agent_id].delta_ndcg
            self.best_max_mean_rating_ratio = np.max(agents[best_agent_id].gnome.mean(axis=1))
            # print(np.max(agents[best_agent_id].gnome.mean(axis=1)))
            found_new_best = True
        max_mean_rating_ratio = np.max(agents[best_agent_id].gnome.mean(axis=1))
        
        if self.tb:
            self.tb.add_scalar('max_fit', max_fit, cur_generation)
            self.tb.add_scalar('max_dhr', max_dhr, cur_generation)
            self.tb.add_scalar('max_ndcg', max_ndcg, cur_generation)
            self.tb.add_scalar('best_max_fit', self.best_max_fit, cur_generation)
            self.tb.add_scalar('%DropInLogLoss', self.best_max_fit / self.baseline_fit, cur_generation)
            self.tb.add_scalar('best_max_hr', self.best_max_hr, cur_generation)
            self.tb.add_scalar('best_max_ndcg', self.best_max_ndcg, cur_generation)

            self.tb.add_scalar('max_pos_ratio', max_mean_rating_ratio, cur_generation)

            # tb.add_scalar('HRPercentDrop', best_max_fit/baseline_fit, cur_generation)
            self.tb.add_scalar('min_fit', min_fit, cur_generation)
            self.tb.add_scalar('mean_fit', mean, cur_generation)
            self.tb.add_scalar('std_fit', std, cur_generation)
            self.tb.add_scalar('pool_size', length, cur_generation)
            self.tb.close()

        t_now = time()
        t_all = (t_now - self.start_time)/(60**2)
        t_fit = (t_now - self.curr_generation_time) / 60
        self.curr_generation_time = time()

        output_log = f"G={cur_generation:2}\tp_size={length}\t"\
                    f"min={min_fit:.4f}\tmax={max_fit:.4f}\tmax_pos_ratio={max_mean_rating_ratio:.4f}\t"\
                    f"best_max={self.best_max_fit:.4f} (G={self.best_max_fit_g})\tbest_max_pos_ratio={self.best_max_mean_rating_ratio:.4f}\t"\
                    f"avg={mean:.4f}\tstd={std:.4f}\tfit[{t_fit:0.2f}m]\tall[{t_all:0.2f}h]"
        logger.info(output_log)
        print(output_log)
        return found_new_best

    @staticmethod
    def get_best_agent(agents):
        sorted(agents, key=lambda x: x.fitness, reverse=True)
        return agents[0].fitness, agents[0].delta_hr, agents[0].delta_ndcg


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

    @staticmethod
    def save(agents, n_fake_users, train_frac, cur_generation, selection, save_dir='agents'):
        import pickle
        import os
        # g={cur_generation}
        with open(os.path.join(save_dir, f'agents_m={selection}_dump_n_fake={n_fake_users}_t={train_frac}.dmp'), 'wb') as file:
            pickle.dump(agents, file)
        logger.info('saved in generation={}'.format(cur_generation))


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
    N_ITEMS = 100
    # BINARY = False  # binary or non binary data
    POS_RATIO = 0.01 # Ratio pos/ neg ratio  one percent from each user

    # print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    N_GENERATIONS = 100

    # selection_mode = 'TOURNAMENT' # 'ROULETTE'
    selection_mode = 'ROULETTE'
    crossover_type = 'items'
    crossover_type = 'simple'
    ga_params = {
        'POP_SIZE': POP_SIZE,
        'N_GENERATIONS': N_GENERATIONS,
        'SELECTION_GENERATIONS_BEFORE_REMOVAL': SELECTION_GENERATIONS_BEFORE_REMOVAL,
        'SELECTION_REMOVE_PERCENTILE': SELECTION_REMOVE_PERCENTILE,
        'MUTATE_USER_PROB': MUTATE_USER_PROB,
        'MUTATE_BIT_PROB': MUTATE_BIT_PROB,
        'CONVERT_BINARY': True,
        'POS_RATIO': POS_RATIO,
        'MAX_POS_RATIO': 0.12,
        'CROSSOVER_CREATE_TOP': CROSSOVER_CREATE_TOP,
        'SELECTION_MODE': selection_mode,
        'crossover_type': crossover_type

    }
    ga = FakeUserGeneticAlgorithm(ga_params, tb=False, baseline= 0.001)

    agents = ga.init_agents(n_fake_users=N_FAKE_USERS, n_items= N_ITEMS)
    # tb = SummaryWriter(comment = f'test_{POS_RATIO}_{SELECTION_GENERATIONS_BEFORE_REMOVAL}_{SELECTION_REMOVE_PERCENTILE}_{MUTATE_USER_PROB}_{MUTATE_BIT_PROB}_{CROSSOVER_CREATE_TOP}')
    print('created n_agents', len(agents))
    n_created = 0
    for cur_generation in range(1, N_GENERATIONS):
        agents = ga.fitness(agents)
        # if cur_generation % 3 == 0:
        # ga.get_stats_writer(agents, cur_generation, tb)
        ga.get_stats_writer(agents, cur_generation)
        # if cur_generation > 50:
        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)







# pop = [agent for agent in ]

if __name__ == '__main__':
    main()