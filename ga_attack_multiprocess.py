import sys
import fire
import logging

from ga import FakeUserGeneticAlgorithm
from Evalute import pert_train_evaluate_model
from Data import *
from FitnessProcessPool import FitnessProcessPool
from tensorboardX import SummaryWriter
from Constants import *
from ga_attack_train_baseline import get_weights

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ml1m = 'movielens1m'
ml100k = 'movielens100k'

# HYPER-PARAMETERS
np.random.seed(SEED)

MUTATE_USER_PROB = 0.5  # prob for choosing an individual
MUTATE_BIT_PROB = 0.02  # prob for flipping a bit
### TOURNEMENT
SELECTION_GENERATIONS_BEFORE_REMOVAL = 5
SELECTION_REMOVE_PERCENTILE = 0.05  # remove only worst 5% after they have passed SELECTION_GENERATIONS_BEFORE_REMOVAL
CROSSOVER_CREATE_TOP = 10  # Select top # to create pairs of offsprings.

# Dataset Related
CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1
BASE_MODEL_EPOCHS = 15  # will get the best model out of these n epochs.

# # Attack hyperparams:
# # PERT_MODEL_TAKE_BEST = False
# MODEL_P_EPOCHS = 3 # 3  # Will take best model (in terms of highest HR and NDCG) if MODEL_TAKE_BEST is set to true
# TRAINING_SET_AGENT_FRAC = 0.5  # FRAC of training set for training the model
# POS_RATIO = 0.05  # Ratio pos/ neg ratio  one percent from each user

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

    attack_benign_training_set = concat_and_shuffle(malicious_training_set, train_set)
    best_pert_model, best_pert_hr, best_pert_ndcg = pert_train_evaluate_model(model, attack_benign_training_set,
                                                                              attack_params['test_set'],

                                                                              batch_size=batch_size,
                                                                              epochs=Constants.MODEL_P_EPOCHS,
                                                                              # pert_model_take_best=PERT_MODEL_TAKE_BEST,
                                                                              user_item_matrix_reindexed=attack_params['user_item_matrix_reindexed'],
                                                                              verbose=VERBOSE)
    t5 = time()
    delta_hr = attack_params['best_base_hr'] - best_pert_hr
    delta_ndcg = attack_params['best_base_ndcg'] - best_pert_ndcg
    agent_fitness = max(delta_hr, 0)

    # agent_fitness = (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas
    if VERBOSE:
        beign_malicious_ratio = len(train_set[0]) / len(malicious_training_set[0])
        print(f'id:{agent.id}\tratio:{beign_malicious_ratio:0.2f}\tage:{agent.age}\tΔhr:{delta_hr:0.4f}\tΔndcg:{delta_ndcg:0.4f}\tf:{agent_fitness:0.4f}\ttotal_time={t5-t0:0.1f}s')

    return agent_fitness, delta_ndcg

 # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
def main(n_fake_users=10, selection= 'TOURNAMENT', pop_size= 500, pos_ratio= 0.06, max_pos_ratio=0.12, crossover_type='items',
         train_frac=0.01, n_generations= 100, n_processes= 4, save_dir= 'agents', out_log=True, save=True, dataset=ml100k):
    DATASET_NAME = dataset
    out_params = f'd_{DATASET_NAME}_m{selection}_u{n_fake_users}_pop{pop_size}_c{crossover_type}_t{train_frac}_r{pos_ratio}'
    logger = get_logger(f'exp_{out_params}', out_log)
    # selection = 'ROULETTE' # selection = 'ROULETTE'
    train_set, test_set, n_users, n_movies, user_item_matrix_reindexed = get_train_test_set(DATASET_NAME, CONVERT_BINARY)
    weights_path, best_hr, best_ndcg = get_weights(n_fake_users, DATASET_NAME)
    logger.info(f'Trained Base model: n_real_users={n_users}\tn_movies={n_movies}\nBaseline Metrics: best_hr={best_hr:0.4f}\tbest_ndcg={best_ndcg:0.4f}')

    tb = SummaryWriter(comment=f'---exp_pid={os.getpid()}_{out_params}') if out_log else None

    # crossover_type = 'simple'
    # crossover_type = 'items'
    ga_params = {
        'POP_SIZE': pop_size,
        'N_GENERATIONS': n_generations,
        'SELECTION_GENERATIONS_BEFORE_REMOVAL': SELECTION_GENERATIONS_BEFORE_REMOVAL,
        'SELECTION_REMOVE_PERCENTILE': SELECTION_REMOVE_PERCENTILE,
        'MUTATE_USER_PROB': MUTATE_USER_PROB,
        'MUTATE_BIT_PROB': MUTATE_BIT_PROB,
        'CONVERT_BINARY': CONVERT_BINARY,
        'POS_RATIO': pos_ratio,
        'MAX_POS_RATIO': max_pos_ratio,
        'CROSSOVER_CREATE_TOP': CROSSOVER_CREATE_TOP,
        'SELECTION_MODE': selection,
        'crossover_type': crossover_type
    }
    [logger.info(f'{key}={value}') for key, value in ga_params.items()]

    ga = FakeUserGeneticAlgorithm(ga_params, tb, baseline=best_hr)
    agents = ga.init_agents(n_fake_users, n_movies)

    attack_params = {'n_users': n_users, 'n_movies': n_movies, 'best_base_hr': best_hr, 'best_base_ndcg': best_ndcg,
                     'user_item_matrix_reindexed': user_item_matrix_reindexed,
                     'n_fake_users': n_fake_users, 'test_set': test_set, 'dataset_name': DATASET_NAME, 'convert_binary': True}
    fitness_pool = FitnessProcessPool(attack_params, n_processes)

    logger.info(f" created n_agents={len(agents)} , Training each agent with {train_frac:0.0%} of training set ({int(train_frac * len(train_set[0]))} real training samples)")

    train_set_subset = create_subset(train_set, train_frac, DATASET_NAME, Constants.unique_subset_id)
    # TODO: Look on this: CREATING STATIONARY TRAINING SUBSET - attack may overfit to this particular training')

    for cur_generation in range(1, n_generations + 1):
        # train_set_subset = create_subset(train_set, train_frac=train_frac)
        agents = fitness_pool.fitness(agents, train_set_subset)
        found_new_best = ga.get_stats_writer(agents, cur_generation)

        if found_new_best and save:
            ga.save(agents, n_fake_users, train_frac, cur_generation, selection, save_dir=save_dir)
            dhr, dncdg = ga.get_best_agent(agents)
            with open(f'ga_results/result_{DATASET_NAME}_{selection}_{n_fake_users}_{train_frac}', 'w') as f:
                f.write(f'{dhr},{dncdg}')

        if selection == 'RANDOM':  # random attack - every g the pop will be restarted
            agents = ga.init_agents(n_fake_users, n_movies)
            continue
        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)

    fitness_pool.terminate()
    # Save data related to agents


if __name__ == '__main__':
    fire.Fire(main)
