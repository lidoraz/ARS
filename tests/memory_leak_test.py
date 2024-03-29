import os
import resource

os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
from Models.NeuMF import get_model
from keras.optimizers import Adam

from ga import FakeUserGeneticAlgorithm
from Evalute import baseline_train_evalute_model, pert_train_evaluate_model
from Data import *

import tensorflow as tf
from keras.models import model_from_json

tf.logging.set_verbosity(tf.logging.ERROR)

ml1m = 'movielens1m'
ml100k = 'movielens100k'

BASE_MODEL_DIR = '../base_models'

# from tensorboardX import SummaryWriter

# HYPER-PARAMETERS

from Constants import SEED
np.random.seed(SEED)

#GA Hyperparams:
# POP_SIZE = 100
# MAX_POP_SIZE = 100  # 0 - no limit
N_GENERATIONS = 1000
# Mutation
MUTATE_USER_PROB = 0.5  # prob for choosing an individual
MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
# Selection
SELECTION_GENERATIONS_BEFORE_REMOVAL = 5
SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5%
# Crossover
CROSSOVER_CREATE_TOP = 4  # Select top # to create pairs of offsprings.

# Model / Dataset related
# N_FAKE_USERS = 10

# Dataset Related
CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1
BASE_MODEL_EPOCHS = 3  # will get the best model out of these n epochs.

# Attack hyperparams:
PERT_MODEL_TAKE_BEST = False
MODEL_P_EPOCHS = 3 # 3  # Will take best model (in terms of highest HR and NDCG) if MODEL_TAKE_BEST is set to true
TRAINING_SET_AGENT_FRAC = 0.5  # FRAC of training set for training the model
POS_RATIO = 0.02  # Ratio pos/ neg ratio  one percent from each user

CONCURRENT = 0 # number of workers
# CONCURRENT = multiprocessing.cpu_count()
# CONCURRENT = 0
VERBOSE = 1


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


def load_base_model(n_fake_users, name="default_model"):
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{n_fake_users}_e{BASE_MODEL_EPOCHS}_w.h5'

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.name = name
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
    model = attack_params['model']
    model.set_weights(attack_params['baseline_model_weights']) # must reset weights to baseline each time an agent gets evaluated #TODO: not this
    attack_df = convert_attack_agent_to_input_df(agent)
    malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome,
                                                                 n_users=attack_params['n_users'], num_negatives=4)
    attack_benign_training_set = concat_and_shuffle(malicious_training_set, train_set)
    best_pert_model, best_pert_hr, best_pert_ndcg = pert_train_evaluate_model(model, attack_benign_training_set,
                                                                              attack_params['test_set'],
                                                                              batch_size=batch_size,
                                                                              epochs=MODEL_P_EPOCHS,
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
    # del model_copy
    # tf.reset_default_graph() # TODO: not this
    return agent_fitness
    # return sum(sum(agent.gnome))


def _fitness_concurrent(agents, train_set, attack_params):
    """
    Runs concurrent...
    :param agents:
    :param n_users:
    :param train_set:
    :param test_set:
    :param best_base_hr:
    :param best_base_ndcg:
    :return:
    """
    from concurrent.futures.thread import ThreadPoolExecutor
    import tensorflow as tf
    executor = ThreadPoolExecutor(max_workers=CONCURRENT)

    def eval_fitness_func(agent):
        # raise EnvironmentError('this causes oom problems')
        if not agent.evaluted:
            # workaround for tensorflow, each task creates a new graph
            # with tf.Graph() as graph:
            # tf.Graph().
            with tf.Graph().as_default() as graph:
                with tf.Session(graph=graph) as sess:
                    K.set_session(sess)
                    agent_fitness = __get_fitness(agent, train_set, attack_params)
            # gc.collect()
                            # K.clear_session() # TODO: could be problemmatirc
                            # tf.compat.v1.reset_default_graph()
            # tf.reset_default_graph()  # TODO: THIS FIXES THE PROBLEM
            # graph.close()
            return agent_fitness

        else:
            return agent.fitness

    fitness_list = list(executor.map(eval_fitness_func, agents))
    for idx, agent in enumerate(agents):
        agent.fitness = fitness_list[idx]
        agent.evluated = True
    return agents


def _fitness_single(agents,train_set, attack_params):
    # for agent in tqdm(agents, total=len(agents)):
    for agent in agents:
        if not agent.evaluted:
            agent_fitness = get_fitness_single(agent, train_set, attack_params)
            # tf.reset_default_graph()  # TODO: THIS FIXES THE PROBLEM
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

def main(n_fake_users, pop_size = 50, max_pop_size=100,train_frac=TRAINING_SET_AGENT_FRAC):
    POP_SIZE = pop_size
    print('PARAMS:')
    print('Baseline Model Params:')
    print(f'DATASET_NAME:{DATASET_NAME}, TEST_SET_PERCENTAGE:{TEST_SET_PERCENTAGE}, BASE_MODEL_EPOCHS:{BASE_MODEL_EPOCHS}, CONVERT_BINARY:{CONVERT_BINARY}')
    print(f'GA Hyperparams:')
    print(f'POP_SIZE:{POP_SIZE}, N_GENERATIONS:{N_GENERATIONS}, CROSSOVER_TOP: {CROSSOVER_CREATE_TOP}')
    print(f'MUTATE_USER_PROB:{MUTATE_USER_PROB}, MUTATE_BIT_PROB:{MUTATE_BIT_PROB}')
    print(f'SELECTION_GENERATIONS_BEFORE_REMOVAL:{SELECTION_GENERATIONS_BEFORE_REMOVAL}, SELECTION_REMOVE_PERCENTILE:{SELECTION_REMOVE_PERCENTILE} ')
    print('***ATTACK PARAMS:')
    print(f'n_fake_users:{n_fake_users}, TRAINING_SET_AGENT_FRAC:{TRAINING_SET_AGENT_FRAC},'
          f'POS_RATIO:{POS_RATIO}, MODEL_P_EPOCHS:{MODEL_P_EPOCHS}, PERT_MODEL_TAKE_BEST:{PERT_MODEL_TAKE_BEST}')
    print('CONCURRENT=', CONCURRENT)

    model, weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg = train_base_model(n_fake_users)
    baseline_model_weights = model.get_weights()
    attack_params = {'n_users': n_users, 'n_movies': n_movies, 'best_base_hr': best_hr, 'best_base_ndcg': best_ndcg,
                     'n_fake_user': n_fake_users, 'baseline_model_weights': baseline_model_weights, 'model': model, 'test_set':test_set}
    print(f'Trained Base model:: n_real_users:{n_users}\tn_movies:{n_movies}\tbest_hr:{best_hr:0.4f}\tbest_ndcg:{best_ndcg:0.4f}')
    print("ADVERSRIAL PHASE")

    ga = FakeUserGeneticAlgorithm(POP_SIZE=pop_size,
                                  MAX_POP_SIZE=max_pop_size,
                                  N_GENERATIONS=N_GENERATIONS,
                                  SELECTION_GENERATIONS_BEFORE_REMOVAL=SELECTION_GENERATIONS_BEFORE_REMOVAL,
                                  SELECTION_REMOVE_PERCENTILE=SELECTION_REMOVE_PERCENTILE,
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=CONVERT_BINARY,
                                  POS_RATIO=POS_RATIO,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP)

    agents = ga.init_agents(n_fake_users, n_movies)
    n_new_agents = 0
    print('created n_agents', len(agents))
    # print(f"Training each agent with {TRAINING_SET_AGENT_FRAC:0.0%} of training set ({int(TRAINING_SET_AGENT_FRAC * len(train_set[0]))} real training samples)")
    for cur_generation in range(1, N_GENERATIONS):
        get_fitness_single(agents[0], train_set, attack_params)
        # train_setsubset = create_subset(train_set, train_frac=train_frac) # TODO:  not this
        # agents = fitness(agents,train_set, attack_params)
        # pool_size, min_fit, max_fit, mean, std = ga.get_stats(agents)
        max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10 ** 3)  #linux computes in kbytes, while mac in bytes
        print(f"G={cur_generation}\tmem_usage={max_mem_usage:0.0f} MB")

import fire

if __name__ == '__main__':
    fire.Fire(main)
