from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
from Models.NeuMF import get_model
from Evalute import evaluate_model
from DataLoader import *
from Data import *
from tensorflow.keras.optimizers import Adam

import pandas as pd
from ga import FakeUserGeneticAlgorithm
from GA_Attack.RandomUserAttack_NeuMF import train_evaluate_model, plot
from tensorflow.keras.models import clone_model
ml1m = 'movielens1m'
ml100k = 'movielens100k'



BASE_MODEL_DIR = 'base_models'

# HYPER-PARAMETERS
POP_SIZE = 10
N_GENERATIONS = 100
# Mutation
MUTATE_USER_PROB = 0.2  # prob for choosing an individual
MUTATE_BIT_PROB = 0.01  # prob for flipping a bit
# Selection
GENERATIONS_BEFORE_REMOVAL = 50
REMOVE_PERCENTILE = 0.05  # remove only worst 5%

# Model / Dataset related
N_FAKE_USERS = 3
N_ITEMS = 10
BINARY = False  # binary or non binary data
POS_RATIO = 0.1  # Ratio pos/ neg ratio  one percent from each user


CONVERT_BINARY = True
DATASET_NAME = ml100k
TEST_SET_PERCENTAGE = 1

N_MAL_USERS = 50

BASE_MODEL_EPOCHS = 1  # TODO: CHANGE
MODEL_P_EPOCHS = 3
VERBOSE = 1

def convert_attack_agent_to_input_df(agent):
    users, items = np.nonzero(agent.gnome)
    ratings = agent.gnome[(users, items)]
    df = pd.DataFrame(
        {'user_id': users,
         'item_id': items,
         'rating':ratings})
    return df

# add n_user offset for malicious users
def create_training_instances_malicious(df,  user_item_matrix, n_users, num_negatives= 4):
    user_input, item_input, labels = [], [], []
    negative_items = {user: np.argwhere(user_item_matrix[user]==0).flatten() for user in df['user_id'].unique()}
    for index, row in df.iterrows():
        user = row['user_id']
        user_input.append(user)
        item_input.append(row['item_id'])
        labels.append(row['rating'])
        negative_input_items = np.random.choice(negative_items[user], num_negatives)
        for neg_item in negative_input_items:
            user_input.append(user)
            item_input.append(neg_item)
            labels.append(0)

    training_set = (np.array(user_input) + n_users, np.array(item_input), np.array(labels))
    # print('len(training_set):', len(training_set))
    return training_set

def train_base_model():
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)

    data = Data(seed=42)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)
    # mal_training = fake_user_selected_one_item(n_mal_users, n_users, n_movies, convert_binary,
    #                                            n_random_movies=50)
    n_users_w_mal = n_users + N_MAL_USERS + 1
    print('REGULAR PHASE')
    # NeuMF Parameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    verbose = 0
    loss_func = 'binary_crossentropy'
    if loss_func == 'binary_crossentropy':
        assert CONVERT_BINARY

    model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    print('get_model done')

    # this should be the best model according to the evalute process, in terms of HR and NDCG
    model_base, best_hr, best_ndcg = train_evaluate_model(model, train_set, test_set, batch_size=batch_size, epochs=BASE_MODEL_EPOCHS, verbose=verbose)
    # save model
    model_base.save_weights(f'{BASE_MODEL_DIR}/NeuMF_w.h5')
    model_json = model_base.to_json()
    with open(f'{BASE_MODEL_DIR}/NeuMF.json', "w") as json_file:
        json_file.write(model_json)
    print('Saved model and weights at dir:', BASE_MODEL_DIR)

    return n_users, n_movies, test_set, best_hr, best_ndcg

def compile_model(model):
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')


def load_base_model():

    # model_path = f'{BASE_MODEL_DIR}/NeuMF_{n_users_w_mal}_{n_movies}.json'
    # weights_path = f'{BASE_MODEL_DIR}/NeuMF_w_{n_users_w_mal}_{n_movies}.h5'
    model_path = f'{BASE_MODEL_DIR}/NeuMF.json'
    weights_path = f'{BASE_MODEL_DIR}/NeuMF_w.h5'
    from tensorflow.keras.models import model_from_json
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model


def fitness(agents, n_users, test_set, best_base_hr, best_base_ndcg):
    executor = ThreadPoolExecutor(max_workers=10)

    def asyc_func(agent):
        batch_size = 512
        model_copy = load_base_model()
        compile_model(model_copy)
        attack_df = convert_attack_agent_to_input_df(agent)
        malicious_training_set = create_training_instances_malicious(df=attack_df, user_item_matrix=agent.gnome,
                                                                     n_users=n_users, num_negatives=4)
        best_pert_model, best_hr, best_ndcg = train_evaluate_model(model_copy, malicious_training_set, test_set,
                                                                   batch_size=batch_size,
                                                                   epochs=MODEL_P_EPOCHS, verbose=VERBOSE)
        delta_hr = best_base_hr - best_hr
        delta_ndcg = best_base_ndcg - best_ndcg

        if VERBOSE:
            print(f'agent_id: {agent.id} ; delta_hr: {delta_hr:0.2f} ; delta_ndcg: {delta_ndcg:0.2f}')
        return (2 * delta_hr * delta_ndcg) / (delta_hr + delta_ndcg)  # harmonic mean between deltas

    fitness_list = list(executor.map(asyc_func, agents))
    for agent, agent_fitness in zip(agents, fitness_list):
        agent.fitness = fitness
    return agents

def main():
    # init_model()
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    n_users, n_movies, test_set, best_hr, best_ndcg = train_base_model()
    N_ITEMS = n_movies

    print("ADVERSRIAL PHASE")

    # print(AttackAgent(N_FAKE_USERS, N_ITEMS).gnome)
    ga = FakeUserGeneticAlgorithm()

    agents = ga.init_agents(N_FAKE_USERS, N_ITEMS, POP_SIZE)


    print('created n_agents', len(agents))
    ga.print_stats(agents, 0)
    for cur_generation in range(1, N_GENERATIONS):
        agents = fitness(agents, n_users, test_set, best_hr, best_ndcg)
        if cur_generation % 50 == 0:
            ga.print_stats(agents , cur_generation)

        agents = ga.selection(agents)
        agents = ga.crossover(agents, cur_generation)
        agents = ga.mutation(agents)







# pop = [agent for agent in ]

if __name__ == '__main__':
    main()