import os

from GA_Attack.Data import convert_attack_agent_to_input_df

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Models.NeuMF import get_model
from keras.optimizers import Adam

from GA_Attack.ga import FakeUserGeneticAlgorithm
from GA_Attack.Evalute import train_evaluate_model, plot
from GA_Attack.Data import *
from keras.models import clone_model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
ml1m = 'movielens1m'
ml100k = 'movielens100k'


BASE_MODEL_DIR = 'base_models'

# HYPER-PARAMETERS
POP_SIZE = 100
N_GENERATIONS = 100
# Mutation
MUTATE_USER_PROB = 0.2  # prob for choosing an individual
MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
# Selection
GENERATIONS_BEFORE_REMOVAL = 50
REMOVE_PERCENTILE = 0.05  # remove only worst 5%

# Model / Dataset related
N_FAKE_USERS = 10
# N_ITEMS = 10
BINARY = False  # binary or non binary data
POS_RATIO = 0.1  # Ratio pos/ neg ratio  one percent from each user


CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1

BASE_MODEL_EPOCHS = 5  # TODO: CHANGE
MODEL_P_EPOCHS = 3
VERBOSE = 2
# add n_user offset for malicious users


def train_base_model():
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)

    data = Data(seed=42)
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
    verbose = 1
    loss_func = 'binary_crossentropy'

    model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    print('get_model done')

    # this should be the best model according to the evalute process, in terms of HR and NDCG
    model_base, best_hr, best_ndcg = train_evaluate_model(model, train_set, test_set, batch_size=batch_size,
                                                          epochs=BASE_MODEL_EPOCHS, verbose=verbose)
    return model_base, n_users, n_movies, test_set, best_hr, best_ndcg

def compile_model(model):
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')

def fitness(agents, base_model, n_users, test_set, best_base_hr, best_base_ndcg):

    batch_size = 512
    for agent in tqdm(agents, total=len(agents),   ):
        model_copy = clone_model(base_model)
        model_copy.set_weights(base_model.get_weights())
        compile_model(model_copy)
        attack_df = convert_attack_agent_to_input_df(agent)
        malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome, n_users=n_users, num_negatives=4)
        best_pert_model, best_hr, best_ndcg = train_evaluate_model(model_copy, malicious_training_set, test_set, batch_size=batch_size,
                                                              epochs=MODEL_P_EPOCHS, verbose=VERBOSE)
        delta_hr = best_base_hr - best_hr
        delta_ndcg = best_base_ndcg - best_ndcg

        agent_fitness = (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas
        if VERBOSE:
            print(f'agent_id: {agent.id} age: {agent.age} ; delta_hr: {delta_hr:0.2f} ; delta_ndcg: {delta_ndcg:0.2f} ; fitness: {agent_fitness: 0.3f}')
        agent.fitness = agent_fitness
    return agents

def main():
    # init_model()
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    base_model, n_users, n_movies, test_set, best_hr, best_ndcg = train_base_model()
    N_ITEMS = n_movies

    print("ADVERSRIAL PHASE")

    # print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    ga = FakeUserGeneticAlgorithm(POP_SIZE, N_GENERATIONS, GENERATIONS_BEFORE_REMOVAL, REMOVE_PERCENTILE,
                                  MUTATE_USER_PROB, MUTATE_BIT_PROB, BINARY, POS_RATIO)

    agents = ga.init_agents(N_FAKE_USERS, N_ITEMS)


    print('created n_agents', len(agents))
    ga.print_stats(agents, 0)
    for cur_generation in range(1, N_GENERATIONS):
        agents = fitness(agents, base_model, n_users, test_set, best_hr, best_ndcg)
        if cur_generation % 50 == 0:
            ga.print_stats(agents , cur_generation)

        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)







# pop = [agent for agent in ]

if __name__ == '__main__':
    main()