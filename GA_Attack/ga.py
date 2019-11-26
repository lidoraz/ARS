import numpy as np
import uuid


class AttackAgent:
    def __init__(self, n_m_users=0, n_items=0, gnome=None, d_birth=0 , POS_RATIO=1, BINARY = True):
        if gnome is not None and ( n_items != 0 or n_items != 0 or d_birth != 0 or POS_RATIO != 1):
            raise ValueError('not valid config')
        self.fitness = .0
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
    def __init__(self, POP_SIZE, N_GENERATIONS, GENERATIONS_BEFORE_REMOVAL, REMOVE_PERCENTILE, MUTATE_USER_PROB, MUTATE_BIT_PROB, BINARY, POS_RATIO):
        self.POP_SIZE = POP_SIZE
        self.N_GENERATIONS = N_GENERATIONS
        self.GENERATIONS_BEFORE_REMOVAL = GENERATIONS_BEFORE_REMOVAL
        self.REMOVE_PERCENTILE = REMOVE_PERCENTILE
        self.MUTATE_USER_PROB = MUTATE_USER_PROB
        self.MUTATE_BIT_PROB = MUTATE_BIT_PROB
        self.BINARY = BINARY
        self.POS_RATIO = POS_RATIO

    def init_agents(self, n_m_users, n_items):
        return [AttackAgent(n_m_users, n_items, POS_RATIO=self.POS_RATIO) for _ in range(self.POP_SIZE)]


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

    """
    Sorts the pool, removes old indviduals that are over GENERATIONS_BEFORE_REMOVAL and are worse than REMOVE_PERCENTILE in score
    """

    # TODO: there is a case where the pool can get too large, think about it.
    def selection(self, agents):
        # update age
        for agent in agents:
            agent.age +=1
        # sort by fitness best to worse
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        # get 5% worst
        fitness_treshold = agents[int((1-self.REMOVE_PERCENTILE) * len(agents))].fitness
        # agents_removed_worst = [a for a in agents if a.fitness > fitness_treshold and curr_generation - a.d_birth < GENERATIONS_BEFORE_REMOVAL]

        remove_func = lambda x: x.age < self.GENERATIONS_BEFORE_REMOVAL or x.fitness < fitness_treshold

        agents_removed_worst = list(filter(remove_func, agents))
        return agents_removed_worst

        # sort by fitness, take 2 best agents
        # keep best one in pool

    # TODO: Extend cross-over between pairs
    def crossover(self, agents, cur_generation):
        # Simple Cross-over between 2 agents, creates 2 offsprings.
        # Improve this to have a tournement like.
        agent_1_part_prefix = agents[0].gnome[:agents[0].gnome.shape[0]//2]
        agent_1_part_postfix = agents[0].gnome[agents[0].gnome.shape[0] // 2:]
        agent_2_part_prefix = agents[1].gnome[:agents[1].gnome.shape[0]//2]
        agent_2_part_postfix = agents[1].gnome[agents[1].gnome.shape[0] // 2:]
        offspring_1 = np.concatenate([agent_1_part_prefix, agent_2_part_postfix])
        offspring_2 = np.concatenate([agent_2_part_prefix, agent_1_part_postfix])
        offspring_1 = AttackAgent(gnome=offspring_1, d_birth= cur_generation)
        offspring_2 = AttackAgent(gnome=offspring_2, d_birth= cur_generation)

        # add offsprints and remove worst
        agents.append(offspring_1)
        agents.append(offspring_2)
        return agents
    # return offspring_1, offspring_2

    # Make 2 offsprings from 2 best agents
    # the cross over will take bunch of users from each attack, and mix them up together,
    # the cross over will not change the ratings themselvs, only the rows.
#   # make also with the hall of fame



    def mutation(self, agents):
        # mutation utility functions
        def bit_flip_func_binary(self, x):
            # bits = [1, 0]
            if np.random.rand() < self.MUTATE_BIT_PROB:
                if x == 0:
                    return 1
                else:
                    return 0
            else:
                return x

        def bit_flip_func_non_binary(self, x):
            if np.random.rand() < self.MUTATE_BIT_PROB:
                return np.random.randint(1, 6)
            else:
                return x

        def flip_bit_1d_array(self, arr):
            if self.BINARY:
                return list(map(bit_flip_func_binary, arr))
            else:
                return list(map(bit_flip_func_non_binary, arr))
        for agent in agents:
            if np.random.rand() < self.MUTATE_USER_PROB:
                agent.gnome = np.apply_along_axis(flip_bit_1d_array, 0, agent.gnome)
                agent.generations_mutated += 1
        # flip bit in an entry in a prob
        # this will work on every entry, to create stohastic behaviour, kind of epsilon greedy method.
        return agents

    def print_stats(self, agents, cur_generation):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness for ind in agents]

        length = len(agents)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print(f"G: {cur_generation} ; p_size: {length} ; min: {min(fits):.2f} ; max: {max(fits):.2f} ; avg: {mean:.2f} ; std: {std:.2f}")
        # print(f"Best agent index: {np.argmax(fits)}")

# TODO: When called from outside, still these parameters are used, need to find a way to change these
# TODO: need those parameters from lambda functions.




def main():
    # HYPER-PARAMETERS
    POP_SIZE = 100
    N_GENERATIONS = 1000
    # Mutation
    MUTATE_USER_PROB = 0.2  # prob for choosing an individual
    MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
    # Selection
    GENERATIONS_BEFORE_REMOVAL = 50
    REMOVE_PERCENTILE = 0.05  # remove only worst 5%

    # Model / Dataset related
    N_FAKE_USERS = 9
    N_ITEMS = 7
    BINARY = False  # binary or non binary data
    POS_RATIO = 0.1  # Ratio pos/ neg ratio  one percent from each user

    print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    ga = FakeUserGeneticAlgorithm(POP_SIZE, N_GENERATIONS, GENERATIONS_BEFORE_REMOVAL, REMOVE_PERCENTILE, MUTATE_USER_PROB, MUTATE_BIT_PROB, BINARY, POS_RATIO)

    agents = ga.init_agents(N_FAKE_USERS, N_ITEMS)

    print('created n_agents', len(agents))
    ga.print_stats(agents, 0)
    for cur_generation in range(1, N_GENERATIONS):
        agents = ga.fitness(agents)
        if cur_generation % 50 == 0:
            ga.print_stats(agents , cur_generation)

        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)







# pop = [agent for agent in ]

if __name__ == '__main__':
    main()