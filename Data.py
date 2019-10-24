# loader
from Constants import CONFIG

# generate train test

# evalute using metrics HR and NDCG
# import tensorflow as tf
import pandas as pd
import numpy as np


import os
import numpy as np
# import tensorflow as tf
# import tensorlayer as tl

# import pandas as pd


class Data():
    def __init__(self,
                 df, user_item_matrix, total_users, total_movies
                 , negative_set_size= 99, seed= None):
        self.negative_set_size = negative_set_size
        self.df = df
        self.user_item_matrix = user_item_matrix
        self.total_users = total_users
        self.total_movies = total_movies
        self.most_recent_entries = None

        if not (seed is None):
            self.seed = seed
            np.random.seed(seed)
    def get_user_id_list(self):
        return sorted(self.df['user_id'].unique())
    def get_movie_id_list(self):
        return sorted(self.df['movie_id'].unique())
    """
    Returns a dataframe without most recent entries, that are used for the test set
    """
    def filter_trainset(self):
        self.most_recent_entries = self.df.loc[self.df.groupby('user_id')['timestamp'].idxmax()]
        df_removed_recents = self.df.drop(self.most_recent_entries.index)
        return df_removed_recents


    """
    This function creates negative examples given a user name from a data frame
    For each user, it samples self.negative_set_size indexes, and adding a real rated sample
    In order to by evaluated using HR and NDCG metrics
    :input A (0-1] range number for sampling a percentage of the users
    :return A dict with key='user_id', val=[neg,neg,...,pos]
            where pos is most recent rating by the user
    """
    def create_testset(self, percent=1):
        # select one most recent entry from each user, this will be the test
        if self.most_recent_entries is None:
            self.most_recent_entries = self.df.loc[self.df.groupby('user_id')['timestamp'].idxmax()]

        assert len(self.most_recent_entries) == self.total_users # each user must have exactly one entry in most recent data
        most_recent_entries = self.most_recent_entries.sample(frac=percent)
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

    """
    This function creates negative examples given a user name from a data frame
    :input A user_id from the user_item_matrix (Pivot table created from user_id, item_id, rating)
    :returns: A random list in size "self.negative_set_size" of item_ids that user_id did not interact with
    
    """
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


    #TODO: See where to put this
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

from DataLoader import get_movielens100k

if __name__ == '__main__':
    df, user_item_matrix, total_users, total_movies = get_movielens100k(convert_binary= False)

    data = Data(df, user_item_matrix, total_users, total_movies,seed= 42)
    train_set = data.create_trainset()
    test_set = data.create_testset()
    print(test_set)
