# from DataLoader import *
# from Data import Data
# from Evalute import evaluate_model
# from Models.SimpleCF import SimpleCF
# ml1m = 'movielens1m'
# ml100k = 'movielens100k'
# from time import time
#
# import numpy as np
#
#
# def init_model():  # trained / load
#     pass
#
# def train_evaluate_model(low_rank_cf_model, train, valid, test_set, epochs, n_users, n_movies):
#
#     best_hr = 0
#     best_ndcg = 0
#     low_rank_cf_model.set_model(n_users, n_movies, n_latent_factors=64)
#     t0 = time()
#     mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
#     print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.1f s]'
#           % (mean_hr, mean_ndcg, time() - t0))
#     for epoch in range(epochs):
#         t1 = time()
#         loss = low_rank_cf_model.fit_once(train, valid, batch_size=128, verbose=0)
#         t2 = time()
#         mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
#         t3 = time()
#         print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
#               % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss, t3 - t2))
#         if mean_hr > best_hr and mean_ndcg > best_ndcg:
#             best_hr = mean_hr
#             best_ndcg = mean_ndcg
#             # low_rank_cf_model.save_model()
#     print('Total time: [%.1f s]' % (time() - t0))
#
# def _generate_entries(new_users_entries, n_new_users, n_movies):
#     new_user_ids = [10 ** 7 + user_id for user_id in range(n_new_users)]
#     new_entries = [[10 ** 7 + user_id, movie_id, new_users_entries[user_id, movie_id], -1] for user_id in range(n_new_users) for
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
#
# def fake_user_zeros(n_new_users, n_movies):
#     print('fake_user_zeros()')
#     new_users_entries = np.zeros((n_new_users, n_movies))
#
#     return _generate_entries(new_users_entries, n_new_users, n_movies)
#
# def fake_user_selected_one_item(n_new_users, n_movies, convert_binary, n_random_movies=20):
#     print('fake_user_selected_one_item()')
#     new_users_entries = np.zeros((n_new_users, n_movies))  # assuems binary data
#     random_movie_ids = np.random.choice(list(range(n_movies)), n_random_movies)
#     # random_movie_id = np.random.randint(n_movies)
#     print(f'chosen random_movie_ids:{random_movie_ids}')
#     for random_movie_id in random_movie_ids:
#         if convert_binary:
#
#             new_users_entries[:, random_movie_id] = 1
#         else:
#             new_users_entries[:, random_movie_id] = 5
#     return _generate_entries(new_users_entries, n_new_users, n_movies)
#
# def fake_user_selected_randomitems(n_new_users, n_movies, convert_binary):
#     pass
#
#
# def main():
#     # init_model()
#     # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
#     convert_binary = True
#     load_model = True
#     dataset_name = ml100k
#     testset_percentage = 0.5
#     n_mal_users = 500
#
#
#     epochs = 5
#     print(f'Started... convert_binary={convert_binary}')
#
#     # df, user_item_matrix, total_users, total_movies = get_movielens100k(convert_binary)
#     df, user_item_matrix, total_users, total_movies = get_from_dataset_name(dataset_name, convert_binary)
#     data = Data(df, user_item_matrix, total_users, total_movies, seed=42)
#     df_removed_recents = data.filter_trainset()  # TODO: make logic here simpler
#
#     test_set = data.create_testset(percent=testset_percentage)
#
#     print('Baseline PHASE - added malicious users with no data')
#     new_entries, new_user_ids = fake_user_zeros(n_mal_users, total_movies)
#     df_added = df_removed_recents.append(new_entries)
#     new_user_id_list = np.array(data.get_user_id_list() + new_user_ids)
#     low_rank_cf_model = SimpleCF()
#
#     train, valid, n_users, n_movies = low_rank_cf_model.model_preprocessing(df_added, new_user_id_list, data.get_movie_id_list())
#     #Train and Evaluate
#     train_evaluate_model(low_rank_cf_model, train, valid, test_set, epochs, n_users, n_movies)
#
#     print("ADVERSRIAL PHASE")
#     # get loss
#     low_rank_cf_model.model # get_layer()
#
#
#
#     # new_entries, new_user_ids = fake_user_selected_one_item(n_mal_users, total_movies, convert_binary)
#     # df_added = df_removed_recents.append(new_entries)
#     # new_user_id_list = np.array(data.get_user_id_list() + new_user_ids)
#     #
#     # low_rank_cf_model = SimpleCF()
#     # train, valid, n_users, n_movies = low_rank_cf_model.model_preprocessing(df_added, new_user_id_list,
#     #                                                                         data.get_movie_id_list())
#     # train_evaluate_model(low_rank_cf_model, train, valid, test_set, epochs, n_users, n_movies)
#
# if __name__ == '__main__':
#     main()
