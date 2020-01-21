import sys
import fire
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


def get_logger(logger_name, save_logger):
    import datetime
    logger_name = datetime.datetime.now().strftime('%Y-%m-%d_') + logger_name
    logger = logging.getLogger('ga_attack')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if save_logger:
        fh = logging.FileHandler('logs/' + logger_name)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
    return logger


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
    t0 = time()
    batch_size = 512

    attack_df = convert_attack_agent_to_input_df(agent)
    malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome,
                                                                 n_users=attack_params['n_users'], num_negatives=4)
    # logger = logging.getLogger('ga_attack')
    # n_malicious_examples_include_negatives = len(malicious_training_set[0])
    # logger.info('#Attack_df={} #Entries in malicious dataset - {} (includes negative_sampling={}), Which is {}% of poison from real dataset'
    #             .format(len(attack_df), n_malicious_examples_include_negatives, 4, round((n_malicious_examples_include_negatives / len(train_set[0])) * 100, 2)))
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

 # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
def main(n_fake_users=10, selection= 'TOURNAMENT', pop_size= 500, pos_ratio= 0.02, max_pos_ratio=0.15,
         max_pop_size=1000, train_frac=0.01, n_generations= 100, n_processes= 4, save_dir= 'agents', out_log=True, save=True,
         SELECTION_GENERATIONS_BEFORE_REMOVAL=500):

    logger = get_logger('exp_{}_u{}_pop{}_t{}.log)'.format(selection, n_fake_users, pop_size, train_frac), out_log)
    # selection = 'ROULETTE' # selection = 'ROULETTE'
    weights_path, train_set, test_set, n_users, n_movies, best_hr, best_ndcg = get_baseline_stats(n_fake_users)
    logger.info(f'Trained Base model: n_real_users={n_users}\tn_movies={n_movies}\nBaseline Metrics: best_hr={best_hr:0.4f}\tbest_ndcg={best_ndcg:0.4f}')

    tb = SummaryWriter(comment=f'---exp_pid={os.getpid()}_m{selection}_u{n_fake_users}_pop{pop_size}_t{train_frac}_r{pos_ratio}') if out_log else None

    ga_params = {
        'POP_SIZE': pop_size,
        'MAX_POP_SIZE': max_pop_size,
        'N_GENERATIONS': n_generations,
        'SELECTION_GENERATIONS_BEFORE_REMOVAL': SELECTION_GENERATIONS_BEFORE_REMOVAL,
        'SELECTION_REMOVE_PERCENTILE': SELECTION_REMOVE_PERCENTILE,
        'MUTATE_USER_PROB': MUTATE_USER_PROB,
        'MUTATE_BIT_PROB': MUTATE_BIT_PROB,
        'CONVERT_BINARY': CONVERT_BINARY,
        'POS_RATIO': pos_ratio,
        'MAX_POS_RATIO': max_pos_ratio,
        'CROSSOVER_CREATE_TOP': CROSSOVER_CREATE_TOP,
        'SELECTION_MODE': selection
    }
    [logger.info(f'{key}={value}') for key, value in ga_params.items()]

    ga = FakeUserGeneticAlgorithm(ga_params, tb, baseline=best_hr)
    agents = ga.init_agents(n_fake_users, n_movies)

    attack_params = {'n_users': n_users, 'n_movies': n_movies, 'best_base_hr': best_hr, 'best_base_ndcg': best_ndcg,
                     'n_fake_users': n_fake_users, 'test_set': test_set}
    fitness_pool = FitnessProcessPool(attack_params, n_processes)

    logger.info(f" created n_agents={len(agents)} , Training each agent with {train_frac:0.0%} of training set ({int(train_frac * len(train_set[0]))} real training samples)")


    # train_set_subset = create_subset(train_set, train_frac=train_frac) # TODO: Look on this: CREATING STATIONARY TRAINING SUBSET - attack may overfit to this particular training')

    for cur_generation in range(1, n_generations + 1):
        train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = fitness_pool.fitness(agents, train_set_subset)
        found_new_best = ga.get_stats_writer(agents, cur_generation)

        if found_new_best and save:
            ga.save(agents, n_fake_users, train_frac, cur_generation, selection, save_dir=save_dir)

        if selection == 'RANDOM':  # random attack - every g the pop will be restarted
            agents = ga.init_agents(n_fake_users, n_movies)
            continue
        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)

    fitness_pool.terminate()

if __name__ == '__main__':
    fire.Fire(main)
