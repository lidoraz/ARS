import os
os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from multiprocessing import Process, Queue
from queue import Empty


class FitnessProcessPool:
    @staticmethod
    def process_function(in_agents_queue: Queue, out_fitness_queue, process_status, attack_params):
        print(f'started process... p={os.getpid()}, pp={os.getppid()}')
        from ga_attack_multiprocess import load_base_model, get_fitness_single
        pid = os.getpid()
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        model = load_base_model(attack_params['n_fake_users'])
        model_base_weights = model.get_weights()

        while True:
            try:
                idx, agent, train_set = in_agents_queue.get(block=True)  # block on lock, wait for data
                model.set_weights(model_base_weights)
                agent_fitness = get_fitness_single(agent, train_set, attack_params, model)
                out_fitness_queue.put((idx, agent_fitness))
            except Empty:  # Empty Exception
                process_status.put(pid)
    ################################################

    def __init__(self, attack_params, n_processes=4):
        self.attack_params = attack_params
        self.N_PROCESSES = n_processes
        self.in_agents_queue = Queue()
        self.out_fitness_queue = Queue()
        self.process_status = Queue()
        self.started = False
        self.processes = []

        for i in range(self.N_PROCESSES):
            p = Process(target=self.process_function,
                        args=(self.in_agents_queue, self.out_fitness_queue, self.process_status, self.attack_params))
            p.daemon = True
            self.processes.append(p)
            p.start()

    def fitness(self, agents, training_set):
        for idx, agent in enumerate(agents):  # put data in queue for each process
            self.in_agents_queue.put((idx, agent, training_set))

        for i in range(len(agents)):  # wait on queue for a fitness result
            idx, agent_fitness = self.out_fitness_queue.get(block=True)
            agents[idx].fitness = agent_fitness
        return agents

    def terminate(self):
        for p in self.processes:
            p.terminate()
