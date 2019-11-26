
ml1m = 'movielens1m'
ml100k = 'movielens100k'

from time import time
import numpy as np
import matplotlib.pyplot as plt
from Models.NeuMF import get_model
from GA_Attack.Evalute import evaluate_model
from GA_Attack.DataLoader import *
from GA_Attack.Data import *

from tensorflow.keras.optimizers import Adam
# TODO: Much better to se CPU here instead of GPU, probably because the model is simple,
#  Predict takes 1.5s instead of 44s
from tensorflow.keras.models import clone_model

def train_evaluate_model(model, train_set, test_set, batch_size=256, epochs=5, verbose= 0):
    best_hr = 0
    best_ndcg = 0
    best_epoch = 0
    t0 = time()
    mean_hr, mean_ndcg, _ = evaluate_model(model, test_set, verbose=verbose)
    models = []

    for epoch in range(epochs):
        t1 = time()
        (user_input, item_input, labels) = train_set
        loss = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=verbose > 1, shuffle=True)

        model_copy = clone_model(model)
        model_copy.set_weights(model.get_weights())
        models.append(model_copy)
        t2 = time()
        mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)
        if verbose > 1:
            print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
                  % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss.history['loss'][0], time_eval))
        if mean_hr > best_hr and mean_ndcg > best_ndcg:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            best_epoch = epoch
            # low_rank_cf_model.save_model()
    return models[best_epoch], best_hr, best_ndcg


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
    plt.title(f'Adv Attack: Metrics drop over {(len(HR_list))} epochs')
    ax2 = plt.twinx()
    ax2.set_ylabel('NDCG', color='tab:orange')
    ax2.plot(epochs, NDCG_list, color='orange')
    plt.show()

def main():
    # init_model()
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    convert_binary = True
    load_model = True
    dataset_name = ml100k
    testset_percentage = 1

    n_mal_users = 10


    epochs = 5
    df = get_from_dataset_name(dataset_name, convert_binary)

    data = Data(seed=42)
    train_set, test_set, n_users, n_movies = data.pre_processing(df)
    mal_training = fake_user_selected_one_item(n_mal_users, n_users, n_movies, convert_binary,
                                                                            n_random_movies=50)
    n_users_w_mal = n_users + n_mal_users + 1
    print('REGULAR PHASE')
    # model = SimpleCF()

    mf_dim = 8
    layers = [64,32,16,8]
    reg_layers = [0,0,0,0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    verbose = 0
    loss_func = 'binary_crossentropy'
    if loss_func == 'binary_crossentropy':
        assert convert_binary

    model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    print('get_model done')
    # model.set_model(n_users_w_mal, n_movies, n_latent_factors=64)
    # test_fit(model, train_set, batch_size, epochs=5)
    train_evaluate_model(model, train_set, test_set, batch_size=batch_size, epochs=epochs, verbose=verbose)

    print("ADVERSRIAL PHASE")
    hr_list = []
    ndcg_list = []
    mal_epochs = 150
    for epoch in range(mal_epochs):
        (user_input, item_input, labels) = mal_training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)
        print(f'mal_it: {epoch} HR: {mean_hr:.4f}, NDCG: {mean_ndcg:.4f}, Eval:[{time_eval:0.1f s}')

        # shuffled = np.c_[mal_training[0], mal_training[1], mal_training[2]]
        # np.random.shuffle(shuffled)
        # mal_training = (shuffled[:, 0], shuffled[:, 1], shuffled[:, 2])
        hr_list.append(mean_hr)
        ndcg_list.append(mean_ndcg)

    plot(hr_list, ndcg_list)


if __name__ == '__main__':
    main()




