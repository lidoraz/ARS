from DataLoader import *
from Data import Data
from Evalute import evaluate_model, evaluate_shilling_model
from Models.SimpleCF import SimpleCF
ml1m = 'movielens1m'
ml100k = 'movielens100k'
from time import time

import numpy as np
import matplotlib.pyplot as plt

def init_model():  # trained / load
    pass



def train_evalute_shilling_model(model, train, valid, test_set, shilling_items, epochs):
    t0 = time()
    mean_hr, mean_ndcg = evaluate_model(model, test_set)
    mean_hr_shilling, mean_ndcg_shilling = evaluate_shilling_model(model, test_set, shilling_items)
    print(f'Init: HR = {mean_hr:.4f}, NDCG = {mean_ndcg:.4f}, '
          f'HR_s = {mean_hr_shilling:.4f}, NDCG_s = {mean_ndcg_shilling:.4f} Eval:[{time() - t0:.1f} s]')
    for epoch in range(epochs):
        t1 = time()
        loss = model.fit_once(train, valid, batch_size=128, verbose=0)
        t2 = time()
        mean_hr, mean_ndcg = evaluate_model(model, test_set)
        mean_hr_shilling, mean_ndcg_shilling = evaluate_shilling_model(model, test_set, shilling_items)
        t3 = time()
        print(f'Iteration: {epoch + 1} Fit:[{t2 - t1:.1f} s]: HR = {mean_hr:.4f}, NDCG = {mean_ndcg:.4f}, '
              f'loss = {loss:.4f}, HR_s = {mean_hr_shilling:.4f}, NDCG_s = {mean_ndcg_shilling:.4f} Eval:[{t3 - t2:.1f} s]')
            # low_rank_cf_model.save_model()
    print('Total time: [%.1f s]' % (time() - t0))


# def evalute_model(model, test_set):


def train_evaluate_model(low_rank_cf_model, train_set, test_set, epochs):

    best_hr = 0
    best_ndcg = 0
    t0 = time()
    mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
    print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.1f s]'
          % (mean_hr, mean_ndcg, time() - t0))
    for epoch in range(epochs):
        t1 = time()
        loss = low_rank_cf_model.fit_once(train_set, batch_size=128, verbose=0)
        t2 = time()
        mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
        t3 = time()
        print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
              % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss, t3 - t2))
        if mean_hr > best_hr and mean_ndcg > best_ndcg:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            # low_rank_cf_model.save_model()
    print('Total time: [%.1f s]' % (time() - t0))

# def _generate_entries(new_users_entries, n_new_users, n_users, n_movies):
#     new_user_ids = [n_users + user_id for user_id in range(1, n_new_users)]
#     new_entries = [[n_users + user_id, movie_id, new_users_entries[user_id, movie_id], -1] for user_id in range(1, n_new_users) for
#                    movie_id in range(1, n_movies)]
#     # conv to df way - row is user_id, movie_id, rating, timestamp
#     new_entries = pd.DataFrame(new_entries, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
#     return new_entries, new_user_ids
#
# """
# This seems to help the model increase it's HR and NDCG, could be part of adversarial training ?
# """
# def fake_user_random(n_new_users, n_movies, convert_binary):
#     print('fake_user_random()')
#     if convert_binary:
#         new_users_entries = np.random.randint(0, 2, (n_new_users, n_movies))  # assuems binary data
#     else:
#         new_users_entries = np.random.randint(0, 6, (n_new_users, n_movies))
#
#     return _generate_entries(new_users_entries, n_new_users, n_movies)
#
# def fake_user_zeros(n_new_users, n_movies):
#     print('fake_user_zeros()')
#     new_users_entries = np.zeros((n_new_users, n_movies))
#
#     return _generate_entries(new_users_entries, n_new_users, n_movies)


def fake_user_selected_one_item(n_new_users, n_users, n_movies, convert_binary, n_random_movies=20, negative_samples = 4):
    print('fake_user_selected_one_item()')
    # new_users_entries = np.zeros((n_new_users, n_movies))  # assuems binary data
    random_movie_ids = np.random.choice(list(range(n_movies)), n_random_movies)
    rest_movie_ids = [movie_id for movie_id in range(n_movies) if movie_id not in random_movie_ids]
    user_ids = []
    movie_ids = []
    ratings = []
    for idx in range(1, n_new_users + 1):
        mal_user_id = idx + n_users
        for random_movie_id in random_movie_ids:
            user_ids += [mal_user_id] * (negative_samples + 1)
            movie_ids.append(random_movie_id)
            if convert_binary:
                ratings.append(1)
            else:
                ratings.append(5)
            movie_ids += list(np.random.choice(rest_movie_ids, negative_samples))
            ratings += [0] * negative_samples
    shuffled = np.c_[user_ids, movie_ids, ratings]
    np.random.shuffle(shuffled)

    mal_training = (shuffled[:, 0], shuffled[:, 1], shuffled[:, 2])
    return mal_training

# TODO: THIS WITH NO NEGATIVE SAMPLING
# def fake_user_selected_one_item(n_new_users, n_users, n_movies, convert_binary, n_random_movies=20):
#     print('fake_user_selected_one_item()')
#     # new_users_entries = np.zeros((n_new_users, n_movies))  # assuems binary data
#     random_movie_ids = np.random.choice(list(range(n_movies)), n_random_movies)
#     user_ids = []
#     movie_ids = []
#     ratings = []
#     for idx in range(1, n_new_users):
#         mal_user_id = idx + n_users
#         user_ids += [mal_user_id] * n_random_movies
#         movie_ids += list(random_movie_ids)
#         if convert_binary:
#             ratings += [1] * n_random_movies
#         else:
#             ratings += [5] * n_random_movies
#     mal_training = (np.array(user_ids), np.array(movie_ids), np.array(ratings))
#     return mal_training

def fake_user_selected_randomitems(n_new_users, n_movies, convert_binary):
    pass


def plot(HR_list, NDCG_list):
    epochs = list(range(len(HR_list)))
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, HR_list)
    plt.xlabel('Epoch')
    plt.ylabel('Hit Rate')
    plt.title(f'Adv Attack: Metrics drop over {epochs}')
    ax2 = plt.twinx()
    ax2.set_ylabel('NDCG', color='tab:orange')
    ax2.plot(epochs, NDCG_list, color='orange')
    plt.show()

def main():
    # init_model()
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    convert_binary = False
    load_model = True
    dataset_name = ml100k
    testset_percentage = 0.5

    n_mal_users = 10


    epochs = 5
    print(f'Started... convert_binary={convert_binary}')
    df = get_from_dataset_name(dataset_name, convert_binary)

    print('preparing data..')
    data = Data(df, seed=42)

    train_set, test_set, n_users, n_movies = data.pre_processing()

    mal_training = fake_user_selected_one_item(n_mal_users, n_users, n_movies, convert_binary,
                                                                            n_random_movies=50)
    n_users_w_mal = n_users + n_mal_users
    print('REGULAR PHASE')
    model = SimpleCF()
    model.set_model(n_users_w_mal, n_movies, n_latent_factors=64)
    train_evaluate_model(model, train_set, test_set, epochs)

    print("ADVERSRIAL PHASE")
    hr_list = []
    ndcg_list = []
    mal_epochs = 150
    for epoch in range(mal_epochs):
        model.fit_once(mal_training)
        mean_hr, mean_ndcg = evaluate_model(model, test_set)
        print(f'mal_it: {epoch} HR: {mean_hr:.4f}, NDCG: {mean_ndcg:.4f}')

        shuffled = np.c_[mal_training[0], mal_training[1], mal_training[2]]
        np.random.shuffle(shuffled)
        mal_training = (shuffled[:, 0], shuffled[:, 1], shuffled[:, 2])
        hr_list.append(mean_hr)
        ndcg_list.append(mean_ndcg)

    plot(hr_list, ndcg_list)
    #TODO:
    # Here magically we will learn how to add different rating to the mal users, such that it will increase the loss of the model
    # Two problems: How do we generate input that increases the loss of the model
    # 2nd: how do we make sure that after the model retrains, or atleast, retrains on our *NEW* data, it will achieve poor results?
    # 3rd how do we minimize the amount of #mal_users, and #changed ratings.


    # TODO: Side note for generating a random item: max(movie_id) > len(movie_ids)

    # df_added = train_set.append(new_entries)
    # # new_user_id_list = np.array(data.get_user_id_list() + new_user_ids)
    #
    # data_aug = Data(df_added, seed=42)
    # model = SimpleCF()
    # train_set, test_set, n_users, n_movies, userid2idx, itemid2idx = data_aug.pre_processing()
    # model.set_model(n_users, n_movies, n_latent_factors=64)
    # # train, valid, n_users, n_movies = low_rank_cf_model.model_preprocessing(df_added, new_user_id_list,
    # #                                                                         data.get_movie_id_list())
    # train_evalute_shilling_model(model, train_set, test_set, shilling_items, epochs=epochs)

if __name__ == '__main__':
    # x = fake_user_selected_one_item(10, 500, 500, False, 20)
    main()

    # plot(list(range(100)), list(range(100)))


