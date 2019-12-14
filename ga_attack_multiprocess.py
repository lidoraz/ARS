import resource
import sys
import logging
from ga import FakeUserGeneticAlgorithm
from Evalute import pert_train_evaluate_model
from Data import *
from FitnessProcessPool import FitnessProcessPool
from tensorboardX import SummaryWriter
from Constants import SEED

os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ml1m = 'movielens1m'
ml100k = 'movielens100k'

BASE_MODEL_DIR = 'base_models'

# HYPER-PARAMETERS
np.random.seed(SEED)

MUTATE_USER_PROB = 0.5  # prob for choosing an individual
MUTATE_BIT_PROB = 0.02  # prob for flipping a bit
### TOURNEMENT
SELECTION_GENERATIONS_BEFORE_REMOVAL = 5
SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5% after they have passed SELECTION_GENERATIONS_BEFORE_REMOVAL
CROSSOVER_CREATE_TOP = 7  # Select top # to create pairs of offsprings.

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

VERBOSE = 0 # Verbose: 2 - print all in addition to iteration for each agent.



def get_baseline_stats(n_fake_users):
    import json
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)
    data = Data(seed=SEED)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)

    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={BASE_MODEL_EPOCHS}'
    model_path = f'{BASE_MODEL_DIR}/{params_name}.json'
    metrics_path = f'{BASE_MODEL_DIR}/{params_name}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/{params_name}_w.h5'

    assert os.path.exists(model_path), f'Model does not exists at: {model_path}'
    with open(metrics_path, 'r') as metrics_file:
        # model = load_base_model(n_fake_users)
        metrics = json.load(metrics_file)
        best_hr = metrics['best_hr']
        best_ndcg = metrics['best_ndcg']
    return weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg


def load_base_model(n_fake_users):
    from keras.models import model_from_json
    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={BASE_MODEL_EPOCHS}'
    model_path = f'{BASE_MODEL_DIR}/{params_name}.json'
    weights_path = f'{BASE_MODEL_DIR}/{params_name}_w.h5'
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
    # print(os.getpid(), 'get_fitness_single..')

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
        beign_malicious_ratio = len(train_set[0]) / len(malicious_training_set[0])
        print(f'id:{agent.id}\tratio:{beign_malicious_ratio:0.2f}\tage:{agent.age}\tΔhr:{delta_hr:0.4f}\tΔndcg:{delta_ndcg:0.4f}\tf:{agent_fitness:0.4f}\ttotal_time={t5-t0:0.1f}s')
    return agent_fitness
#
# def run_ga_exp(n_fake_users=10, selection = 'TOURNAMENT', pop_size = 500, max_pop_size=1000,train_frac=0.01, n_generations = 1000, n_processes = 4, save_dir = 'agents', out_log=True):
#
#
#
# def run_random_exp(n_fake_users=10, selection = 'TOURNAMENT', pop_size = 500, max_pop_size=1000,train_frac=0.01, n_generations = 1000, n_processes = 4, save_dir = 'agents', out_log=True):

 # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
def main(n_fake_users=10, selection = 'TOURNAMENT', pop_size = 500, max_pop_size=1000,train_frac=0.01, n_generations = 1000, n_processes = 4, save_dir = 'agents', out_log=True):

    # selection = 'ROULETTE' # selection = 'ROULETTE'
    weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg = get_baseline_stats(n_fake_users)
    logger = logging.getLogger('ga_attack')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler(f'logs/exp_{selection}_u{n_fake_users}_pop{pop_size}_t{train_frac}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
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
                                  MUTATE_USER_PROB=MUTATE_USER_PROB,
                                  MUTATE_BIT_PROB=MUTATE_BIT_PROB,
                                  CONVERT_BINARY=CONVERT_BINARY,
                                  POS_RATIO=POS_RATIO,
                                  CROSSOVER_CREATE_TOP=CROSSOVER_CREATE_TOP,
                                  SELECTION_MODE=selection)

    agents = ga.init_agents(n_fake_users, n_movies)
    fitness_pool = FitnessProcessPool(attack_params, n_processes)

    logger.info(f" created n_agents={len(agents)} , Training each agent with {train_frac:0.0%} of training set ({int(train_frac * len(train_set[0]))} real training samples)")

    t0 = time()
    # train_set_subset = create_subset(train_set, train_frac=train_frac) # TODO: Look on this: CREATING STATIONARY TRAINING SUBSET - attack may overfit to this particular training')
    tb = SummaryWriter(comment=f'---exp_pid={os.getpid()}_m{selection}_u{n_fake_users}_pop{pop_size}_t{train_frac}') if out_log else None
    best_max_fit = 0
    best_max_fit_g = 0
    baseline_fit = best_hr
    for cur_generation in range(1, n_generations):
        t1 = time()
        train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = fitness_pool.fitness(agents, train_set_subset)
        t2 = time() - t1
        t4 = (time() - t0) / 60
        pool_size, min_fit, max_fit, mean, std = ga.get_stats_writer(agents, cur_generation, best_max_fit, baseline_fit, tb)
        if max_fit > best_max_fit:
            best_max_fit_g = cur_generation
            best_max_fit = max_fit
            if out_log:
                ga.save(agents, n_fake_users, train_frac, cur_generation, selection, save_dir=save_dir)
        max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10 ** 6)   #linux computes in kbytes, while mac in bytes
        logger.info(f"G={cur_generation}\tp_size={pool_size}\t"
              f"min={min_fit:.4f}\tmax={max_fit:.4f}\t best_max={best_max_fit:.4f}(G={best_max_fit_g})\t"
              f"avg={mean:.4f}\tstd={std:.4f}\tfit[{t2:0.2f}s]\t"
              f"all[{t4:0.2f}m]\tmem_usage={max_mem_usage: 0.3} GB")

        if selection == 'RANDOM':  # random attack - every g the pop will be restarted
            agents = ga.init_agents(n_fake_users, n_movies)
            continue
        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)

import fire

if __name__ == '__main__':
    fire.Fire(main)
