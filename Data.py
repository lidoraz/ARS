# generate train test

# import tensorflow as tf


import os
import numpy as np
from tqdm import tqdm

# import tensorflow as tf
# import tensorlayer as tl

import pickle
# import pandas as pd


"""
This class takes the loaded movie_lens DataFrame and generates:
* Training set
* Test Set
* Utility functions user_id / movie_id to index
"""

class Data():
    def __init__(self, negative_set_size=99, seed=None):
        if not (seed is None):
            self.seed = seed
            np.random.seed(seed)
        self.negative_set_size = negative_set_size
        self.most_recent_entries = None
        self._userid2idx = None
        self._itemid2idx = None

    @staticmethod
    def shuffle_training(traning_set):
        shuffled = np.c_[traning_set[0], traning_set[1], traning_set[2]]
        np.random.shuffle(shuffled)
        traning_set = (shuffled[:, 0], shuffled[:, 1], shuffled[:, 2])
        return traning_set

    @staticmethod
    def _create_negative_items(df, negative_items_path):
        print(f"Could not find '{negative_items_path}', creating...")
        negative_items = {}
        users = list(sorted(df['user_id'].unique()))
        for idx, user in tqdm(enumerate(users), total=len(users)):
            negative_items[user] = df[df['user_id'] != user]['movie_id'].unique()
        pickle.dump(negative_items, open(negative_items_path, "wb"))
        return negative_items


    @staticmethod
    def _create_training_instances(df, negative_items, num_negatives, training_instances_path, percent):
        print(f"Could not find '{training_instances_path}', creating...")
        df = df.groupby('user_id').apply(lambda s: s.sample(frac=percent))
        user_input, item_input, labels = [], [], []
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            user = row['user_id']
            user_input.append(user)
            item_input.append(row['movie_id'])
            labels.append(row['rating'])
            negative_input_items = np.random.choice(negative_items[user], num_negatives)
            for neg_item in negative_input_items:
                user_input.append(user)
                item_input.append(neg_item)
                labels.append(0)

        training_set = (np.array(user_input), np.array(item_input), np.array(labels))
        pickle.dump(training_set, open(training_instances_path, "wb"))
        return training_set

    # for every user, in generates 1 correct input, and num_negatives incorrect inputs
    def get_train_instances(self, df, num_negatives, percent = 1.0):
        rating_type = 'binary_rating' if self.binary else 'multi_rating'
        train_dir_path = os.path.join(os.path.curdir, 'training_dump')
        negative_items_path = os.path.join(train_dir_path, f'ml_neg_{self.n_rows_df}_{rating_type}.p')
        training_instances_path = os.path.join(train_dir_path, f'ml_train_{self.n_rows_df}_{rating_type}.p')
        if os.path.exists(training_instances_path):
            return pickle.load(open(training_instances_path, 'rb'))
        else:
            if os.path.exists(negative_items_path):
                negative_items = pickle.load(open(negative_items_path, 'rb'))
            else:
                negative_items = self._create_negative_items(df, negative_items_path)
            return self._create_training_instances(df, negative_items, num_negatives, training_instances_path, percent)

    # split preprocessing to train and test.
    # Returns n_users & n_movies + 1 each to avoid 0 to 1 issues.
    def pre_processing(self, df, test_percent=1.0, train_precent= 1.0):
        print('pre_processing.. ')
        self.binary = True if len(df['rating'].unique()) == 1 else False
        self.n_rows_df = len(df)
        # creates a id->idx mapping, both user and item
        # reindex the dataframe in order to avoid problems with missing movie_ids
        self._userid2idx = {o: i for i, o in enumerate(df['user_id'].unique())}
        self._itemid2idx = {o: i for i, o in enumerate(df['movie_id'].unique())}
        df_reindexed = df.copy()
        df_reindexed['user_id'] = df_reindexed['user_id'].apply(lambda x: self._userid2idx[x])
        df_reindexed['movie_id'] = df_reindexed['movie_id'].apply(lambda x: self._itemid2idx[x])
        self.n_users = df_reindexed['user_id'].max() + 1
        self.n_movies = df_reindexed['movie_id'].max() + 1
        print('n_users:', self.n_users, 'n_movies:', self.n_movies)
        # TODO: look where to put this
        self.user_item_matrix_reindexed = pd.pivot_table(data=df_reindexed, values='rating', index='user_id', columns='movie_id').fillna(0)
        df_reindexed_removed_recents, most_recent_entries = self._filter_trainset(df_reindexed)
        training_set = self.get_train_instances(df_reindexed_removed_recents, num_negatives=4, percent=train_precent)
        test_set = self._create_testset(df_reindexed, most_recent_entries, test_percent)
        print('train_set size (/w negative sampling):', len(training_set[0]), 'test_set: size', len(test_set))
        return training_set, test_set, self.n_users, self.n_movies

    """
    Returns a dataframe without most recent entries, that are used for the test set
    """
    def _filter_trainset(self, df_reindexed):
        most_recent_entries = df_reindexed.loc[df_reindexed.groupby('user_id')['timestamp'].idxmax()]
        assert len(most_recent_entries) == self.n_users # each user must have exactly one entry in most recent data;
        df_reindexed_removed_recents = df_reindexed.drop(most_recent_entries.index)
        return df_reindexed_removed_recents, most_recent_entries

    """
    This function creates negative examples given a user name from a data frame
    For each user, it samples self.negative_set_size indexes, and adding a real rated sample
    In order to by evaluated using HR and NDCG metrics
    :input A (0-1] range number for sampling a percentage of the users
    :return A dict with key='user_id', val=[neg,neg,...,pos]
            where pos is most recent rating by the user
    """
    def _create_testset(self, df, most_recent_entries, percent=1.0):
        # select one most recent entry from each user, this will be the test
        most_recent_entries = most_recent_entries.sample(frac=percent)
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
        current_user = self.user_item_matrix_reindexed.iloc[user_id]
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

    #
    #
    # def get_movielens_reshaped_GAN(self, util_df, total_users, batch_size, image_size=64):
    #     users_data = np.zeros((total_users, 64, 64), dtype=np.float32)
    #     for idx, user_row in enumerate(util_df.iterrows()):
    #         # user_row = np.zeros((64*64))
    #         user_row_values = user_row[1].values
    #         leading_zeros = image_size ** 2 - len(user_row_values)
    #         user_row_values_scaled = np.append(user_row_values, [0] * leading_zeros)
    #         users_data[idx] = user_row_values_scaled.reshape((image_size, image_size))
    #
    #     # normalize data
    #     users_data = (users_data - 2.5) / 2.5
    #
    #     users_data = np.expand_dims(users_data, 4)
    #
    #     train_ds = tf.data.Dataset.from_tensor_slices(users_data)
    #     # train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    #     ds = train_ds.shuffle(buffer_size=4096)
    #     # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    #     # ds = ds.repeat(n_epoch)
    #     # ds = ds.map(_map_fn, num_parallel_calls=4)
    #     ds = ds.batch(batch_size)
    #     ds = ds.prefetch(buffer_size=2)
    #     return ds, users_data.shape[0]

from DataLoader import *
from time import time

def create_datasets():
    print('creating train datasets for faster debugging')
    dataframes = [get_movielens1m(convert_binary=False),
                  get_movielens100k(convert_binary=False),
                  get_movielens1m(convert_binary=True),
                  get_movielens100k(convert_binary=True)]
    for df in dataframes:
        data = Data(seed=42)
        t0 = time()
        training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=1, train_precent=1)
        print(f'data.pre_processing done. T:{time() - t0}')
        user_input, item_input, labels = training_set
        print(user_input[:10], item_input[:10], labels[:10])

def test_Data_pre_processing_dataset():
    df = get_movielens100k(convert_binary=False)
    data = Data(seed=42)
    t0 = time()
    training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=1, train_precent=1)
    print(f'data.pre_processing done. T:{time() - t0}')
    user_input, item_input, labels = training_set
    print(user_input[:10], item_input[:10], labels[:10])


if __name__ == '__main__':
    create_datasets()
