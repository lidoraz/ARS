import os
import resource
import multiprocessing
os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from keras import backend as K
from NeuMF import get_model
from keras.optimizers import Adam

from ga import FakeUserGeneticAlgorithm
from Evalute import baseline_train_evalute_model, pert_train_evaluate_model\
    # , plot
from Data import *

import tensorflow as tf
from keras.models import model_from_json

tf.logging.set_verbosity(tf.logging.ERROR)

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
    if not os.path.exists(model_path):
        print(model_path, 'does not exists.. creating a baseline model')
        print('REGULAR PHASE')
        model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
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
    else:
        print('Model exists, loading..')
        with open(metrics_path, 'r') as metrics_file:
            model = load_base_model(n_fake_users)
            metrics = json.load(metrics_file)
            best_hr = metrics['best_hr']
            best_ndcg = metrics['best_ndcg']
    return model, weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg

def load_base_model(n_fake_users):
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_w.h5'

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    return model

def get_fitness_single(agent, train_set, attack_params):
    """
    Main function used to create attack per agent
    Cannot run concurrently
    Notes: pert_train_evaluate_model takes 95% of the time for this function, rest of the functions here are minor
    """
    t0 = time()
    batch_size = 512
    # model = load_base_model(attack_params['n_fake_users'])
    model = attack_params['model']
    model.set_weights(attack_params['baseline_model_weights']) # must reset weights to baseline each time an agent gets evaluated
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
    agent_fitness = delta_hr
    # agent_fitness = (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas
    if VERBOSE:
        # print(f'id={agent.id} total_time={t5-t0:0.2f}s'
        #       f' load_model={t1-t0:0.2f}s convert={t2-t1:0.2f}s'
        #       f' create={t3-t2:0.2f}s concat={t4-t3:0.2f}s per_train_eval={t5-t4:0.2f}s')

        beign_malicious_ratio = len(train_set[0]) / len(malicious_training_set[0])
        print(f'id:{agent.id}\tratio:{beign_malicious_ratio:0.2f}\tage:{agent.age}\tΔhr:{delta_hr:0.4f}\tΔndcg:{delta_ndcg:0.4f}\tf:{agent_fitness:0.4f}\ttotal_time={t5-t0:0.1f}s')
    return agent_fitness
    # return sum(sum(agent.gnome))


# TODO: Thread pool will not be idieal here, process pool or something like that might be better with shared resources.
# def _fitness_concurrent(agents, train_set, attack_params):
#     """
#     Runs concurrent...
#     :param agents:
#     :param n_users:
#     :param train_set:
#     :param test_set:
#     :param best_base_hr:
#     :param best_base_ndcg:
#     :return:
#     """
#     from concurrent.futures.thread import ThreadPoolExecutor
#     import tensorflow as tf
#     executor = ThreadPoolExecutor(max_workers=CONCURRENT)
#
#     def eval_fitness_func(agent):
#         raise EnvironmentError('this causes oom problems')
#         if not agent.evaluted:
#             # workaround for tensorflow, each task creates a new graph
#             # with tf.Graph() as graph:
#             # tf.Graph().
#             with tf.Graph().as_default() as graph:
#                 with tf.Session(graph=graph) as sess:
#                     K.set_session(sess)
#                     agent_fitness = __get_fitness(agent, train_set, attack_params)
#             # gc.collect()
#                             # K.clear_session() # TODO: could be problemmatirc
#                             # tf.compat.v1.reset_default_graph()
#             # tf.reset_default_graph()  # TODO: THIS FIXES THE PROBLEM
#             # graph.close()
#             return agent_fitness
#
#         else:
#             return agent.fitness
#
#     fitness_list = list(executor.map(eval_fitness_func, agents))
#     for idx, agent in enumerate(agents):
#         agent.fitness = fitness_list[idx]
#         agent.evluated = True
#     return agents


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
def main(n_fake_users, train_frac=0.01, n_generations = 1000, pos_ratio = 0.02, pop_size = 2000, save_dir = 'agents'):
    logger = logging.getLogger('ga_attack')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'logs/random_exp_u={n_fake_users}_t={train_frac}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.info('PARAMS')
    logger.info('*Baseline Model Params*')
    logger.info(f'DATASET_NAME={DATASET_NAME}, TEST_SET_PERCENTAGE={TEST_SET_PERCENTAGE},'
          f' BASE_MODEL_EPOCHS={BASE_MODEL_EPOCHS}, CONVERT_BINARY={CONVERT_BINARY}')
    logger.info('***ATTACK PARAMS***')

    logger.info(f'n_fake_users={n_fake_users}, TRAINING_SET_AGENT_FRAC={TRAINING_SET_AGENT_FRAC}')
    logger.info(f'POS_RATIO={POS_RATIO}, MODEL_P_EPOCHS={MODEL_P_EPOCHS}')

    model, weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg = train_base_model(n_fake_users)
    baseline_model_weights = model.get_weights()
    attack_params = {'n_users': n_users, 'n_movies': n_movies, 'best_base_hr': best_hr, 'best_base_ndcg': best_ndcg,
                     'n_fake_users': n_fake_users, 'test_set': test_set,
                     'baseline_model_weights': baseline_model_weights, 'model': model,
                     }
    logger.info(f'Trained Base model: n_real_users={n_users}\tn_movies={n_movies}\t'
          f'Baseline Metrics: best_hr={best_hr:0.4f}\tbest_ndcg={best_ndcg:0.4f}')
    logger.info("ADVERSRIAL PHASE")


    ga = FakeUserGeneticAlgorithm(POP_SIZE=pop_size,
                                  MAX_POP_SIZE=0,
                                  N_GENERATIONS=n_generations,
                                  #TODO remove these when using roulette
                                  SELECTION_GENERATIONS_BEFORE_REMOVAL=SELECTION_GENERATIONS_BEFORE_REMOVAL,
                                  SELECTION_REMOVE_PERCENTILE=SELECTION_REMOVE_PERCENTILE,
                                  # up here
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=CONVERT_BINARY,
                                  POS_RATIO=pos_ratio,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP,
                                  SELECTION_MODE=0)


    t0 = time()
    ##### Logging
    # TODO: Look on this: CREATING STATIONARY TRAINING SUBSET - attack may overfit to this particular training')
    # train_set_subset = create_subset(train_set, train_frac=train_frac)
    tb = SummaryWriter(comment=f'-random_exp__u={n_fake_users}_t={train_frac}')
    BEST_MAX_FIT_RANDOM = 0
    for cur_generation in range(1, n_generations):
        t1 = time()
        agents = ga.init_agents(n_fake_users, n_movies)
        train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = fitness(agents,train_set_subset, attack_params)
        t2 = time() - t1
        t4 = (time() - t0) / 60
        pool_size, min_fit, max_fit, mean, std = ga.get_stats_writer_random(agents, cur_generation, tb, BEST_MAX_FIT_RANDOM)
        if max_fit > BEST_MAX_FIT_RANDOM:
            BEST_MAX_FIT_RANDOM = max_fit
        max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10 ** 6)  #linux computes in kbytes, while mac in bytes
        logger.info(f"G={cur_generation}\tp_size={pool_size}\tcreated={CROSSOVER_CREATE_TOP* (CROSSOVER_CREATE_TOP-1)}\tmin={min_fit:.4f}\tmax={max_fit:.4f}\t"
              f"avg={mean:.4f}\tstd={std:.4f}\t"f"fit[{t2:0.2f}s]\t"
              f"all[{t4:0.2f}m]\tmem_usage={max_mem_usage: 0.3} GB")

        # print(f'G:{cur_generation}\tfitness_:[{t1:0.2f}s]\toverall_time:[{t2:0.2f}s]\telapsed:[{((time() - t0_s) / 60):0.2f}m]')
import fire

if __name__ == '__main__':
    fire.Fire(main)
