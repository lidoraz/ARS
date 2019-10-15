# loader
from Constants import CONFIG

# generate train test

# evalute using metrics HR and NDCG
import tensorflow as tf
import pandas as pd
import numpy as np


import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

import pandas as pd

## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)


class Data():
    def __init__(self, dataset, negative_set_size = 99, seed = None):
        self.dataset = dataset
        self.negative_set_size = negative_set_size
        if not (seed is None):
            self.seed = seed
            np.random.seed(seed)
    # def get_data(self, dataset = 'MOVIELENS_100k_PATH'):
    #     self.dataset = dataset
    #     if self.dataset == 'MOVIELENS_100k_PATH':
    #         get_movielens100k

    """
    This function creates negative examples given a user name from a data frame
    For each user, it samples self.negative_set_size indexes, and adding a real rated sample
    In order to by evaluated using HR and NDCG metrics
    """
    def create_testset(self, df: pd.DataFrame):
        # select one most recent entry from each user, this will be the test
        most_recent_entries = df.loc[df.groupby('user_id')['timestamp'].idxmax()]
        assert len(most_recent_entries) == self.total_users # each user must have exactly one entry in most recent data
        users_list = most_recent_entries['user_id'].values
        rated_item_list = most_recent_entries['movie_id'].values
        test_set = {}
        for i, user_id in enumerate(users_list):  # range(len(users_list))
            sampled_indexes = self.sample_indexes(i)
            test_set[user_id] = sampled_indexes + [rated_item_list[i]]
        # will be picked up randomly:
        return test_set
    # def create_negatives(self, user_id, user_item_matrix, seed):
    #     pass

    def sample_indexes(self, user_id):
        current_user = self.user_item_matrix.iloc[user_id]
        unrated_current_user = current_user[current_user == 0].index # take unrated items as negative
        sampled_indexes = np.random.choice(unrated_current_user, self.negative_set_size)
        return list(sampled_indexes)

    # this function gets two largest 2 integers where their product equals num
    def find_max_multipicatns(self, num):
        for i in range(2, num // 2):  # Iterate from 2 to n / 2
            if (num % i) == 0:
                break
        else:
            raise ValueError('Got prime number')

        for i in range(num):
            for j in range(i):  # j is always lower than i
                if i * j == num:
                    return i, j


    def get_movielens100k(self):

        df = pd.read_csv(CONFIG[self.dataset], delimiter='\t', header=None,
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])
        util_df = pd.pivot_table(data=df, values='rating', index='user_id', columns='movie_id').fillna(0)

        total_users = util_df.shape[0]
        total_movies = util_df.shape[1]

        self.df = df
        self.user_item_matrix = util_df
        self.total_users = total_users
        self.total_items = total_movies
        return df, util_df, total_users, total_movies
        # w, h = find_max_multipicatns(total_movies)


    def get_movielens_reshaped_GAN(self, util_df, total_users, batch_size, image_size=64):
        users_data = np.zeros((total_users, 64, 64), dtype=np.float32)
        for idx, user_row in enumerate(util_df.iterrows()):
            # user_row = np.zeros((64*64))
            user_row_values = user_row[1].values
            leading_zeros = image_size ** 2 - len(user_row_values)
            user_row_values_scaled = np.append(user_row_values, [0] * leading_zeros)
            users_data[idx] = user_row_values_scaled.reshape((image_size, image_size))

        # normalize data
        users_data = (users_data - 2.5) / 2.5

        users_data = np.expand_dims(users_data, 4)

        train_ds = tf.data.Dataset.from_tensor_slices(users_data)
        # train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
        ds = train_ds.shuffle(buffer_size=4096)
        # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
        # ds = ds.repeat(n_epoch)
        # ds = ds.map(_map_fn, num_parallel_calls=4)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=2)
        return ds, users_data.shape[0]

if __name__ == '__main__':
    data = Data(dataset= 'MOVIELENS_100k_PATH', seed= 42)
    df, user_item_matrix, total_users, total_movies = data.get_movielens100k()
    test_set = data.create_testset(df)
