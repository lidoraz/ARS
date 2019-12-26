import os
os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from multiprocessing import Process, Queue
from queue import Empty
import logging
import signal

logger = logging.getLogger('ga_attack')


class FitnessProcessPool:
    @staticmethod
    def process_function(in_agents_queue: Queue, out_fitness_queue, process_status, attack_params):
        logger.info(f'pp={os.getppid()} has started process... p={os.getpid()}')
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
        self.to_terminate_process_count = 0

        for i in range(self.N_PROCESSES):
            self.add_start_process()

        signal.signal(signal.SIGTERM, self.receive_signal_term)
        signal.signal(signal.SIGCONT, self.receive_signal_cont)
        signal.signal(signal.SIGALRM, self.receive_signal_stop)
        signal.signal(signal.SIGUSR1, self.receive_signal_add_process)
        signal.signal(signal.SIGUSR2, self.receive_signal_remove_process)
    #

    def add_start_process(self):
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

        while self.to_terminate_process_count > 0:
            self.processes[0].terminate()
            self.to_terminate_process_count -= 1

        return agents

    def terminate(self):
        logger.info('terminate initiated')
        for p in self.processes:
            p.terminate()

        for idx, p in enumerate(self.processes):
            logger.info('waiting for process id: {}/{}...'.format(idx + 1, len(self.processes)))
            p.join()
            logger.info('Process id: {}/{} finished'.format(idx + 1, len(self.processes)))
        logger.info('terminate finished successfully')

    def receive_signal_stop(self, signalNumber, frame):
        logger.info('Received: SIGSTOP')
        for p in self.processes:
            os.kill(p.pid, signal.SIGSTOP)

    def receive_signal_cont(self, signalNumber, frame):
        logger.info('Received: SIGCONT')
        for p in self.processes:
            os.kill(p.pid, signal.SIGCONT)

    def receive_signal_term(self, signalNumber, frame):
        logger.info('Received: SIGTERM')
        self.terminate()
        exit(2)

    def receive_signal_add_process(self, signalNumber, frame):
        logger.info('Received: SIGADD')
        self.add_start_process()

    def receive_signal_remove_process(self, signalNumber, frame):
        logger.info('Received: SIGREMOVE')
        if self.to_terminate_process_count < len(self.processes):
            self.to_terminate_process_count += 1
    #
    #
    # def remove_processor(self):
    #     os.kill(self.processes[0].pid, signal.SIGUSR1)
    #    # harder..
    #    # need to send a signal to the process, so when it finishes the a run, i'll kill itself
    #    # then, need to rmobe it from the process list, but only after the job is completed,
    #    # workaroumd for this is to be connected with the fitness function, that it will read all. and this function here will only change a flag,
    #     # later, on fitness, the process 0 will be removed from the list for good.
    #     #  יכול להיות מצב שעבודה שהתבצעה עבור יוזר ציעלם בגלל שהפרוסס נהרג והטבלה עודכנה לפי שהערך הזה נכנס.
    #     # p = Process(target=self.process_function,
    #     #             args=(self.in_agents_queue, self.out_fitness_queue, self.process_status, self.attack_params))
    #     # p.daemon = True
    #     # self.processes.append(p)
    #     # p.start()
    #     pass