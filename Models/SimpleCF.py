import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

"""
This model is a simple implementation of a Collaborative Filtering.
User table and item table represented with an embedding layer.
Together their dot product has the approximate value to the user_item matrix itself.

"""
checkpoint_cf_dir = "Models/checkpoint_cf"  # "Directory name to save the checkpoints [checkpoint]")


def plot_loss(history):
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 5
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], 'g')
    plt.plot(history.history['val_loss'], 'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()


class SimpleCF:
    def __init__(self):
        self.model = None

    def load_model(self, model_path=f'{checkpoint_cf_dir}/CF.json',
                   weights_path=f'{checkpoint_cf_dir}/CF_w.h5'):
        from tensorflow.keras.models import model_from_json
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights_path)
        print('loaded model from:', model_path, 'weights_path:', weights_path)

    # def model_preprocessing(self, df_removed_recents, sorted_unique_users, sorted_unique_movies):
    #     # self.df_removed_recents = df_removed_recents
    #     self.n_users = len(sorted_unique_users)
    #     self.n_movies = len(sorted_unique_movies)
    #     print('n_users:', self.n_users, 'n_movies:', self.n_movies)
    #     df_dropped = df_removed_recents.copy() # fix for inplace change
    #
    #     # creates a id->idx mapping, both user and item
    #     self._userid2idx = {o: i for i, o in enumerate(sorted_unique_users)}
    #     self._itemid2idx = {o: i for i, o in enumerate(sorted_unique_movies)}
    #
    #     df_dropped['user_id'] = df_dropped['user_id'].apply(lambda x: self._userid2idx[x])
    #     df_dropped['movie_id'] = df_dropped['movie_id'].apply(lambda x: self._itemid2idx[x])
    #     split = np.random.rand(len(df_dropped)) < 0.8
    #     train = df_dropped[split]
    #     valid = df_dropped[~split]
    #
    #     print(train.shape, valid.shape)
    #     return train, valid, self.n_users, self.n_movies

    # includes model.compile with Adam optimizer, loss is MSE
    def set_model(self, n_users, n_movies, n_latent_factors=64):
        n_movies = n_movies + 1
        n_users = n_users + 1
        self.model_postfix = f'{n_users}_{n_movies}'
        user_input = Input(shape=(1,), name='user_input', dtype='int64')
        user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)
        movie_input = Input(shape=(1,), name='movie_input', dtype='int64')
        movie_embedding = Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

        sim = dot([user_vec, movie_vec], name='Simalarity-Dot-Product', axes=1)
        model = Model([user_input, movie_input], sim)

        print('model has been set')
        print(model.summary())
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        self.model = model
        return model

    def fit(self, train, batch_size=128, epochs=25, verbose= 0 ):
        (user_input, item_input, labels) = train

        history = self.model.fit([user_input, item_input], labels, batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose)
        return history


    def fit_once(self, train, batch_size=128, verbose= 0):
        (user_input, item_input, labels) = train

        history = self.model.fit([user_input, item_input], labels, batch_size=batch_size,
                                 epochs=1,
                                 verbose=verbose)
        return history.history['loss'][0]

    def save_model(self):
        # self.model.save(f'{checkpoint_cf_dir}/CF.h5', True, True, 'h5')
        self.model.save_weights(f'{checkpoint_cf_dir}/CF_w.h5')
        model_json = self.model.to_json()
        with open(f'{checkpoint_cf_dir}/CF.json', "w") as json_file:
            json_file.write(model_json)
        print('Saved model and weights at dir:', checkpoint_cf_dir)

    # def create_rating_matrix(self):
    # This method does not work, there are some missing values out there, the new version does this
    #     user_mat = self.model.get_layer('user_embedding').get_weights()[0]
    #     item_mat = self.model.get_layer('movie_embedding').get_weights()[0]
    #     print('user_mat.shape', user_mat.shape)
    #     print('item_mat.shape', item_mat.shape)
    #     assert user_mat.shape[0] == self.n_users and item_mat.shape[0] == self.n_movies
    #     self.RATING_MATRIX = np.dot(user_mat, item_mat.T)

    """
    This function returns the apporximated user_movie to rating matrix 
    Calc may take time. (88s for movieLens 1m)
    """
    # TODO: fix this
    def create_ratings_matrix(self):
        raise NotImplemented
        # self._userid2idx.values()
        # self._itemid2idx.values()
        # import itertools
        # all_pairs = list(itertools.product(self._userid2idx.values(), self._itemid2idx.values()))
        # list1, list2 = zip(*all_pairs)
        # list1 = np.array(list1).reshape(-1,)
        # list2 = np.array(list1).reshape(-1, )
        # ratings_matrix = np.zeros((self.n_users, self.n_movies))
        # res = self.model.predict([list1, list2])
        # for idx,(i, j) in enumerate(all_pairs):
        #     ratings_matrix[i,j] = res[idx]
        # self.RATING_MATRIX = ratings_matrix

    def predict(self, users_items, batch_size, verbose):
        users = users_items[0]
        items = users_items[1]

        assert len(users) == len(items)
        if self.model is None:
            raise EnvironmentError('Cannot predict while model is not init')
        users = np.array(users).reshape(-1,)
        items = np.array(items).reshape(-1,)
        return self.model.predict([users, items]).reshape(-1,)

        # if self.RATING_MATRIX is None:
        #     print('Creating RATING_MATRIX..')
        #     self.create_rating_matrix()
        #     print('Finished Creating RATING_MATRIX..')
        #
        # predictions = []
        # for idx in range(len(users)):
        #     # We use the user_id -> index in order to get the right place in the matrix for corresponding user
        #     predictions.append(self.RATING_MATRIX[self._userid2idx[users[idx]], self._itemid2idx[items[idx]]])
        # return np.array(predictions)

from DataLoader import *
from Data import Data
from Evalute import evaluate_model

from time import time
def main():
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    convert_binary = True
    load_model = False
    testset_percentage = 0.2

    epochs = 10
    print('Started...')

    df = get_movielens1m(convert_binary=False)

    data = Data(df, seed=42)
    t0 = time()
    training_set, test_set, n_users, n_movies = data.pre_processing(test_percent=0.5, train_precent=1)

    low_rank_cf_model = SimpleCF()

    best_hr=0
    best_ndcg = 0
    low_rank_cf_model.set_model(n_users, n_movies, n_latent_factors=64)
    t0 = time()
    mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
    print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.1f s]'
          % (mean_hr, mean_ndcg, time()-t0 ))
    for epoch in range(epochs):
        t1 = time()
        loss = low_rank_cf_model.fit_once(training_set, batch_size=128, verbose=0)
        t2 = time()
        mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)
        t3 = time()
        print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
              % (epoch+1, t2 - t1, mean_hr, mean_ndcg, loss, t3 - t2))
        if mean_hr > best_hr and mean_ndcg > best_ndcg:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            low_rank_cf_model.save_model()
    print('Total time: [%.1f s]' % (time()- t0))


if __name__ == '__main__':
    import os
    main()
