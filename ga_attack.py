import os
import multiprocessing
import resource
from Data import convert_attack_agent_to_input_df

os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
from NeuMF import get_model
from keras.optimizers import Adam

from ga import FakeUserGeneticAlgorithm
from Evalute import baseline_train_evalute_model, pert_train_evaluate_model, plot
from Data import *

import gc
import tensorflow as tf
from keras.models import model_from_json

tf.logging.set_verbosity(tf.logging.ERROR)

ml1m = 'movielens1m'
ml100k = 'movielens100k'

BASE_MODEL_DIR = 'base_models'

# HYPER-PARAMETERS

from Constants import SEED
np.random.seed(SEED)

#GA Hyperparams:
POP_SIZE = 100
MAX_POP_SIZE = 100  # 0 - no limit
N_GENERATIONS = 1000
# Mutation
MUTATE_USER_PROB = 0.5  # prob for choosing an individual
MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
# Selection
SELECTION_GENERATIONS_BEFORE_REMOVAL = 5
SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5%
# Crossover
CROSSOVER_CREATE_TOP = 2  # Select top # to create pairs of offsprings. (will create 3*2 new)

# Model / Dataset related
N_FAKE_USERS = 10
# N_ITEMS = -1 # initiated later



# Dataset Related
CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1
BASE_MODEL_EPOCHS = 15  # will get the best model out of these n epochs.

# Attack hyperparams:
PERT_MODEL_TAKE_BEST = False
MODEL_P_EPOCHS = 3 # 3  # Will take best model (in terms of highest HR and NDCG) if MODEL_TAKE_BEST is set to true
TRAINING_SET_AGENT_FRAC = 0.5  # FRAC of training set for training the model
POS_RATIO = 0.02  # Ratio pos/ neg ratio  one percent from each user

CONCURRENT = 0 # number of workers
# CONCURRENT = multiprocessing.cpu_count()
# CONCURRENT = 0
# VERBOSE = 1
VERBOSE = 0

np.random.seed(42)
# Verbose: 2 - print all in addition to iteration for each agent.
import json

print('CONCURRENT=', CONCURRENT)

def train_base_model():
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)

    data = Data(seed=SEED)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)
    # mal_training = fake_user_selected_one_item(n_mal_users, n_users, n_movies, convert_binary,
    #                                            n_random_movies=50)
    n_users_w_mal = n_users + N_FAKE_USERS + 1
    print('REGULAR PHASE')
    # NeuMF Parameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    loss_func = 'binary_crossentropy'

    model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    print('get_model done')

    # this should be the best model according to the evalute process, in terms of HR and NDCG
    # if not os.path.exists(f'{BASE_MODEL_DIR}/NeuMF.json'):

    print('baseline_train_evalute_model done')
    # save model
    # model_base.save(f'{BASE_MODEL_DIR}/NeuMF_{BASE_MODEL_EPOCHS}_epochs.h5')
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}.json'
    metrics_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}_w.h5'
    if not os.path.exists(model_path):
        model_base, best_hr, best_ndcg = baseline_train_evalute_model(model, train_set, test_set, batch_size=batch_size,
                                                                      epochs=BASE_MODEL_EPOCHS)
        model_json = model_base.to_json()
        model_base.save_weights(weights_path)
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        with open(metrics_path, 'w') as metrics_file:
            json.dump({'best_hr': best_hr, 'best_ndcg': best_ndcg}, metrics_file)
        print('Saved model and weights at dir:', BASE_MODEL_DIR)
    else:
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)
            best_hr = metrics['best_hr']
            best_ndcg = metrics['best_ndcg']
# else:
    print('Model exists')
    return n_users, n_movies, train_set, test_set, best_hr, best_ndcg


from keras.models import load_model


def load_base_model():
    model_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}.json'
    metrics_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_u{N_FAKE_USERS}_e{BASE_MODEL_EPOCHS}_w.h5'

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    return model

def __get_fitness(agent, n_users, train_set, test_set, best_base_hr, best_base_ndcg):
    """
    Main function used to create attack per agent
    Can also run concurrently
    Notes: pert_train_evaluate_model takes 95% of the time for this function, rest of the functions here are minor
    """
    t0 = time()
    batch_size = 512
    model_copy = load_base_model()
    t1 = time()
    attack_df = convert_attack_agent_to_input_df(agent)
    t2 = time()
    malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome,
                                                                 n_users=n_users, num_negatives=4)
    t3 = time()
    attack_benign_training_set = concat_and_shuffle(malicious_training_set, train_set)
    t4 = time()
    best_pert_model, best_pert_hr, best_pert_ndcg = pert_train_evaluate_model(model_copy, attack_benign_training_set,
                                                                              test_set,
                                                                              batch_size=batch_size,
                                                                              epochs=MODEL_P_EPOCHS,
                                                                              pert_model_take_best=PERT_MODEL_TAKE_BEST,
                                                                              verbose=VERBOSE)
    t5 = time()
    delta_hr = best_base_hr - best_pert_hr
    delta_ndcg = best_base_ndcg - best_pert_ndcg
    agent_fitness = delta_hr
    # agent_fitness = (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas
    if VERBOSE:
        # print(f'id={agent.id} total_time={t5-t0:0.2f}s'
        #       f' load_model={t1-t0:0.2f}s convert={t2-t1:0.2f}s'
        #       f' create={t3-t2:0.2f}s concat={t4-t3:0.2f}s per_train_eval={t5-t4:0.2f}s')

        beign_malicious_ratio = len(train_set[0]) / len(malicious_training_set[0])
        print(f'id:{agent.id}\tratio:{beign_malicious_ratio:0.2f}\tage:{agent.age}\tΔhr:{delta_hr:0.4f}\tΔndcg:{delta_ndcg:0.4f}\tf:{agent_fitness:0.4f}\ttotal_time={t5-t0:0.1f}s')
    # del model_copy
    # tf.reset_default_graph() # TODO: THIS FIXES THE PROBLEM
    return agent_fitness
    # return sum(sum(agent.gnome))


def _fitness_concurrent(agents, n_users, train_set, test_set, best_base_hr, best_base_ndcg):
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
        if not agent.evaluted:
            # workaround for tensorflow, each task creates a new graph
            # with tf.Graph() as graph:
            # tf.Graph().
            with tf.Graph().as_default() as graph:
                with tf.Session(graph=graph) as sess:
                    K.set_session(sess)
                    agent_fitness = __get_fitness(agent, n_users, train_set, test_set, best_base_hr, best_base_ndcg)
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


def _fitness_single(agents, n_users, train_set, test_set, best_base_hr, best_base_ndcg):
    # for agent in tqdm(agents, total=len(agents)):
    for agent in agents:
        if not agent.evaluted:
            agent_fitness = __get_fitness(agent, n_users, train_set, test_set, best_base_hr, best_base_ndcg)
            tf.reset_default_graph()  # TODO: THIS FIXES THE PROBLEM
            agent.fitness = agent_fitness
    return agents

#
def fitness(agents, n_users, train_set, test_set, best_base_hr, best_base_ndcg):
    if CONCURRENT:
        return _fitness_concurrent(agents, n_users, train_set, test_set, best_base_hr, best_base_ndcg)
    else:
        return _fitness_single(agents, n_users, train_set, test_set, best_base_hr, best_base_ndcg)

 # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
def main(n_fake_users, pop_size = POP_SIZE, max_pop_size=MAX_POP_SIZE,train_frac=TRAINING_SET_AGENT_FRAC):
    global N_FAKE_USERS
    N_FAKE_USERS = n_fake_users
    print('n_fake_users=',n_fake_users)
    print('pop_size=',pop_size)
    print('max_pop_size=',max_pop_size)
    n_users, n_movies, train_set, test_set, best_hr, best_ndcg = train_base_model()
    K.clear_session()
    tf.reset_default_graph()
    print(f'Trained Base model:: n_users:{n_users}\tn_movies:{n_movies}\tbest_hr:{best_hr:0.4f}\tbest_ndcg:{best_ndcg:0.4f}')
    N_ITEMS = n_movies

    print("ADVERSRIAL PHASE")
    # print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    ga = FakeUserGeneticAlgorithm(POP_SIZE=POP_SIZE,
                                  MAX_POP_SIZE=MAX_POP_SIZE,
                                  N_GENERATIONS=N_GENERATIONS,
                                  SELECTION_GENERATIONS_BEFORE_REMOVAL=SELECTION_GENERATIONS_BEFORE_REMOVAL,
                                  SELECTION_REMOVE_PERCENTILE=SELECTION_REMOVE_PERCENTILE,
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=CONVERT_BINARY,
                                  POS_RATIO=POS_RATIO,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP)

    agents = ga.init_agents(n_fake_users, N_ITEMS)
    n_new_agents = 0
    print('created n_agents', len(agents))
    print(f"Training each agent with {TRAINING_SET_AGENT_FRAC:0.0%} of training set ({int(TRAINING_SET_AGENT_FRAC * len(train_set[0]))} real training samples)")
    t0 = time()
    for cur_generation in range(1, N_GENERATIONS):
        t1 = time()
        train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = fitness(agents, n_users, train_set_subset, test_set, best_hr, best_ndcg)
        t2 = time() - t1
        t4 = (time() - t0) / 60
        pool_size, min_fit, max_fit, mean, std = ga.get_stats(agents)
        max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10 ** 6)  #linux computes in kbytes, while mac in bytes
        print(f"G={cur_generation}\tp_size={pool_size}\tcreated={n_new_agents}\tmin={min_fit:.4f}\tmax={max_fit:.4f}\t"
              f"avg={mean:.4f}\tstd={std:.4f}\t"f"fit[{t2:0.2f}s]\t"
              f"all[{t4:0.2f}m]\tmem_usage={max_mem_usage: 0.3} GB")

        if cur_generation % 100 == 0:
            ga.save(agents, cur_generation)

        agents = ga.selection(agents)
        agents, n_new_agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)
    ga.save(agents,N_FAKE_USERS, N_GENERATIONS)
        # print(f'G:{cur_generation}\tfitness_:[{t1:0.2f}s]\toverall_time:[{t2:0.2f}s]\telapsed:[{((time() - t0_s) / 60):0.2f}m]')
import fire

if __name__ == '__main__':
    fire.Fire(main)
