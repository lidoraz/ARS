import os
import resource


os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from multiprocessing import Process, Queue
from queue import Empty
import sys


def process_function(in_agents_queue: Queue, out_fitness_queue, train_set, attack_params):
    print('started process...', os.getpid(), os.getppid())
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    model = load_base_model(attack_params['n_fake_users'])
    model_base_weights = model.get_weights()
    while True:
        try:
            idx, agent  = in_agents_queue.get(block=True, timeout=3)  # block on lock for 3 sec
            model.set_weights(model_base_weights)
            agent_fitness = get_fitness_single(agent, train_set, attack_params, model)
            out_fitness_queue.put((idx, agent_fitness))
        except Empty: # Empty Exception
            print('got Empty Exception...', os.getpid())
            break



def multiprocess_fitness(agents, training_set, attack_params, N_PROCESSES = 4):
    print('Started multiprocess_fitness')
    in_agents_queue = Queue()
    out_fitness_queue = Queue()

    for idx, agent in enumerate(agents):
        in_agents_queue.put((idx, agent))
    processes = []

    for i in range(N_PROCESSES):
        p = Process(target=process_function, args=(in_agents_queue, out_fitness_queue, training_set, attack_params))
        p.daemon = True
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print('multiprocess_fitness finished')
    while not out_fitness_queue.empty():
        idx, agent_fitness = out_fitness_queue.get()
        agents[idx].fitness = agent_fitness

    return agents
    # TODO: return out_fitness_queue
    # for i in range n_available_proccesses
        # init process i
        # p.start()
    # for i in range n_available_proccesses
        #p.join()
    # fitness has been finished for this generation.







import logging
from NeuMF import get_model


from ga import FakeUserGeneticAlgorithm
from Evalute import baseline_train_evalute_model, pert_train_evaluate_model
    # , plot
from Data import *




# tf.logging.set_verbosity(tf.logging.ERROR)

ml1m = 'movielens1m'
ml100k = 'movielens100k'

BASE_MODEL_DIR = 'base_models'

from tensorboardX import SummaryWriter

# HYPER-PARAMETERS

from Constants import SEED
np.random.seed(SEED)

#GA Hyperparams:
# POP_SIZE = 100
# MAX_POP_SIZE = 100  # 0 - no limit
# N_GENERATIONS = 1000
# Mutation
MUTATE_USER_PROB = 0.5  # prob for choosing an individual
MUTATE_BIT_PROB = 0.02  # prob for flipping a bit
# Selection
SELECTION_GENERATIONS_BEFORE_REMOVAL = 10
SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5% after they have passed SELECTION_GENERATIONS_BEFORE_REMOVAL
# Crossover
CROSSOVER_CREATE_TOP = 7  # Select top # to create pairs of offsprings.

# Model / Dataset related
# N_FAKE_USERS = 10

# Dataset Related
CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1
BASE_MODEL_EPOCHS = 15  # will get the best model out of these n epochs.

# Attack hyperparams:
# PERT_MODEL_TAKE_BEST = False
MODEL_P_EPOCHS = 3 # 3  # Will take best model (in terms of highest HR and NDCG) if MODEL_TAKE_BEST is set to true
TRAINING_SET_AGENT_FRAC = 0.5  # FRAC of training set for training the model
POS_RATIO = 0.02  # Ratio pos/ neg ratio  one percent from each user

CONCURRENT = 0 # number of workers
# CONCURRENT = multiprocessing.cpu_count()
# CONCURRENT = 0
VERBOSE = 0


np.random.seed(42)
# Verbose: 2 - print all in addition to iteration for each agent.
import json



def get_base_stats(n_fake_users):
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)
    data = Data(seed=SEED)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)
    # n_users_w_mal = n_users + n_fake_users + 1
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}.json'
    metrics_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_w.h5'

    assert os.path.exists(model_path)
    print('Model exists, loading..')
    with open(metrics_path, 'r') as metrics_file:
        # model = load_base_model(n_fake_users)
        metrics = json.load(metrics_file)
        best_hr = metrics['best_hr']
        best_ndcg = metrics['best_ndcg']
    return weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg

def train_base_model(n_fake_users):
    """
    this should be the best model according to the evalute process, in terms of HR and NDCG
    """
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)
    data = Data(seed=SEED)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)
    n_users_w_mal = n_users + n_fake_users + 1

    # NeuMF Parameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    loss_func = 'binary_crossentropy'

    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}.json'
    metrics_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_w.h5'
    # if not os.path.exists(model_path):

    print(model_path, 'does not exists.. creating a baseline model')
    print('REGULAR PHASE')
    model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    print('get_model done')
    model_base, best_hr, best_ndcg = baseline_train_evalute_model(model, train_set, test_set, batch_size=batch_size,
                                                                  epochs=BASE_MODEL_EPOCHS)
    print('baseline_train_evalute_model done')
    model_json = model_base.to_json()
    model_base.save_weights(weights_path)
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    with open(metrics_path, 'w') as metrics_file:
        json.dump({'best_hr': best_hr, 'best_ndcg': best_ndcg}, metrics_file)
    print('Saved model and weights at dir:', BASE_MODEL_DIR)

def load_base_model(n_fake_users):
    from keras.models import model_from_json
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_w.h5'
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    return model

def get_fitness_single(agent, train_set, attack_params, model):
    """
    Main function used to create attack per agent
    Cannot run concurrently
    Notes: pert_train_evaluate_model takes 95% of the time for this function, rest of the functions here are minor
    """
    # import time
    # time.sleep(1)
    # agent_fitness = np.random.rand()

    t0 = time()
    batch_size = 512

    attack_df = convert_attack_agent_to_input_df(agent)
    malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome,
                                                                 n_users=attack_params['n_users'], num_negatives=4)
    attack_benign_training_set = concat_and_shuffle(malicious_training_set, train_set)
    best_pert_model, best_pert_hr, best_pert_ndcg = pert_train_evaluate_model(model, attack_benign_training_set,
                                                                              attack_params['test_set'],
                                                                              batch_size=batch_size,
                                                                              epochs=MODEL_P_EPOCHS,
                                                                              # pert_model_take_best=PERT_MODEL_TAKE_BEST,
                                                                              verbose=VERBOSE)
    t5 = time()
    delta_hr = attack_params['best_base_hr'] - best_pert_hr
    delta_ndcg = attack_params['best_base_ndcg'] - best_pert_ndcg
    agent_fitness = max(delta_hr, 0)

    # agent_fitness = (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas
    if VERBOSE:
        # print(f'id={agent.id} total_time={t5-t0:0.2f}s'
        #       f' load_model={t1-t0:0.2f}s convert={t2-t1:0.2f}s'
        #       f' create={t3-t2:0.2f}s concat={t4-t3:0.2f}s per_train_eval={t5-t4:0.2f}s')

        beign_malicious_ratio = len(train_set[0]) / len(malicious_training_set[0])
        print(f'id:{agent.id}\tratio:{beign_malicious_ratio:0.2f}\tage:{agent.age}\tΔhr:{delta_hr:0.4f}\tΔndcg:{delta_ndcg:0.4f}\tf:{agent_fitness:0.4f}\ttotal_time={t5-t0:0.1f}s')
    return agent_fitness
    # return sum(sum(agent.gnome))

def _fitness_single(agents,train_set, attack_params):
    # for agent in tqdm(agents, total=len(agents)):
    for agent in agents:
        if not agent.evaluted:
            agent_fitness = get_fitness_single(agent, train_set, attack_params)
            # tf.reset_default_graph()
            # agent.evaluted = True TODO: what to do here when sub-training set changes on new generation? should evalute every generation?
            agent.fitness = agent_fitness
    return agents

#
def fitness(agents,train_set_subset, attack_params):
    if CONCURRENT:
        pass
        # return _fitness_concurrent(agents,train_set_subset, attack_params)
    else:
        return _fitness_single(agents,train_set_subset, attack_params)

 # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
def main(mode, n_fake_users, pop_size = 500, max_pop_size=100,train_frac=0.01, n_generations = 1000, n_processes = 4, save_dir = 'agents',):

    # TODO: Must train model in another process before.
    if mode == 'TRAIN':
        train_base_model(n_fake_users)
        exit(0)
    assert mode == 'ATTACK' , 'not supported'

    weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg = get_base_stats(n_fake_users)
    logger = logging.getLogger('ga_attack')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'logs/exp_u{n_fake_users}_pop{pop_size}_t{train_frac}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(fh)
    logger.info('PARAMS')
    logger.info('*Baseline Model Params*')
    logger.info(f'DATASET_NAME={DATASET_NAME}, max_pop_size={max_pop_size} TEST_SET_PERCENTAGE={TEST_SET_PERCENTAGE},'
                f' BASE_MODEL_EPOCHS={BASE_MODEL_EPOCHS}, CONVERT_BINARY={CONVERT_BINARY}')
    logger.info('**GA Hyperparams**')
    logger.info(f'POP_SIZE={pop_size}, N_GENERATIONS={n_generations}, CROSSOVER_TOP= {CROSSOVER_CREATE_TOP}')
    logger.info(f'MUTATE_USER_PROB={MUTATE_USER_PROB}, MUTATE_BIT_PROB={MUTATE_BIT_PROB}')
    logger.info(
        f'SELECTION_GENERATIONS_BEFORE_REMOVAL={SELECTION_GENERATIONS_BEFORE_REMOVAL}, SELECTION_REMOVE_PERCENTILE={SELECTION_REMOVE_PERCENTILE} ')
    logger.info('***ATTACK PARAMS***')

    logger.info(f'n_fake_users={n_fake_users}, TRAINING_SET_AGENT_FRAC={TRAINING_SET_AGENT_FRAC}')
    logger.info(f'POS_RATIO={POS_RATIO}, MODEL_P_EPOCHS={MODEL_P_EPOCHS}')

    attack_params = {'n_users': n_users, 'n_movies': n_movies, 'best_base_hr': best_hr, 'best_base_ndcg': best_ndcg,
                     'n_fake_users': n_fake_users, 'test_set': test_set,
                     # 'baseline_model_weights': baseline_model_weights, 'model': model,
                     }
    logger.info(f'Trained Base model: n_real_users={n_users}\tn_movies={n_movies}\t'
          f'Baseline Metrics: best_hr={best_hr:0.4f}\tbest_ndcg={best_ndcg:0.4f}')
    logger.info("ADVERSRIAL PHASE")


    ga = FakeUserGeneticAlgorithm(POP_SIZE=pop_size,
                                  MAX_POP_SIZE=max_pop_size,
                                  N_GENERATIONS=n_generations,

                                  SELECTION_GENERATIONS_BEFORE_REMOVAL=SELECTION_GENERATIONS_BEFORE_REMOVAL,
                                  SELECTION_REMOVE_PERCENTILE=SELECTION_REMOVE_PERCENTILE,
                                  # up here
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=CONVERT_BINARY,
                                  POS_RATIO=POS_RATIO,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP,
                                  SELECTION_MODE='ROULETTE')

    agents = ga.init_agents(n_fake_users, n_movies)
    logger.info(f" created n_agents={len(agents)} , Training each agent with {train_frac:0.0%} of training set ({int(train_frac * len(train_set[0]))} real training samples)")
    t0 = time()
    # TODO: Look on this: CREATING STATIONARY TRAINING SUBSET - attack may overfit to this particular training')
    # train_set_subset = create_subset(train_set, train_frac=train_frac)
    tb = SummaryWriter(comment=f'---exp_pid={os.getpid()}_u{n_fake_users}_pop{pop_size}_t{train_frac}')
    best_max_fit = 0
    best_max_fit_g = 0
    for cur_generation in range(1, n_generations):
        t1 = time()
        train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = multiprocess_fitness(agents, train_set_subset, attack_params, n_processes)
        t2 = time() - t1
        t4 = (time() - t0) / 60
        pool_size, min_fit, max_fit, mean, std = ga.get_stats_writer(agents, cur_generation, best_max_fit, tb)
        if max_fit > best_max_fit:
            best_max_fit_g = cur_generation
            best_max_fit = max_fit
            ga.save(agents, n_fake_users, train_frac, cur_generation, save_dir=save_dir)
        max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10 ** 6)  #linux computes in kbytes, while mac in bytes
        logger.info(f"G={cur_generation}\tp_size={pool_size}\tcreated={CROSSOVER_CREATE_TOP* (CROSSOVER_CREATE_TOP-1)}\t"
              f"min={min_fit:.4f}\tmax={max_fit:.4f}\t best_max={best_max_fit:.4f}(G={best_max_fit_g})\t"
              f"avg={mean:.4f}\tstd={std:.4f}\tfit[{t2:0.2f}s]\t"
              f"all[{t4:0.2f}m]\tmem_usage={max_mem_usage: 0.3} GB")

        agents = ga.selection(agents)
        agents, n_new_agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)

    # ga.save(agents, n_fake_users, train_frac, save_dir=save_dir)
        # print(f'G:{cur_generation}\tfitness_:[{t1:0.2f}s]\toverall_time:[{t2:0.2f}s]\telapsed:[{((time() - t0_s) / 60):0.2f}m]')
import fire

if __name__ == '__main__':
    fire.Fire(main)
