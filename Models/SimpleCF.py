import numpy as np
import pandas as pd


from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.activations import sigmoid
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

    @staticmethod
    def get_model(n_users, n_movies, n_latent_factors=64):
        # self.model_postfix = f'{n_users}_{n_movies}'
        user_input = Input(shape=(None,), name='user_input', dtype='int64')
        user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)
        movie_input = Input(shape=(None,), name='movie_input', dtype='int64')
        movie_embedding = Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='FlattenMovies')(movie_embedding)
        sim = dot([user_vec, movie_vec], name='Simalarity-Dot-Product', axes=1)
        # sim = sigmoid(sim)
        model = Model([user_input, movie_input], sim)

        print('model has been set')
        print(model.summary())
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        return model

    # def fit(self, x, y, batch_size=512, epochs=25, verbose=0):
    #     (user_input, item_input) = x
    #     history = self.model.fit([user_input, item_input], y, batch_size=batch_size,
    #                              epochs=epochs,
    #                              verbose=verbose)
    #     return history
    #
    # def fit_once(self, train, batch_size=128, verbose= 0):
    #     (user_input, item_input, labels) = train
    #
    #     history = self.model.fit([user_input, item_input], labels, batch_size=batch_size,
    #                              epochs=1,
    #                              verbose=verbose)
    #     return history.history['loss'][0]

    # def save_model(self):
    #     # self.model.save(f'{checkpoint_cf_dir}/CF.h5', True, True, 'h5')
    #     self.model.save_weights(f'{checkpoint_cf_dir}/CF_w.h5')
    #     model_json = self.model.to_json()
    #     with open(f'{checkpoint_cf_dir}/CF.json', "w") as json_file:
    #         json_file.write(model_json)
    #     print('Saved model and weights at dir:', checkpoint_cf_dir)
    #
    # def predict(self, users_items, batch_size=512, verbose=0):
    #     users = users_items[0]
    #     items = users_items[1]
    #
    #     assert len(users) == len(items)
    #     if self.model is None:
    #         raise EnvironmentError('Cannot predict while model is not init')
    #     users = np.array(users).reshape(-1,)
    #     items = np.array(items).reshape(-1,)
    #     return self.model.predict([users, items]).reshape(-1,)

from DataLoader import *
from Data import Data
from Evalute import evaluate_model

from time import time
def main():
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    convert_binary = True
    load_model = False
    save_model = False
    testset_percentage = 0.2

    epochs = 10
    print('Started...')

    df = get_movielens100k(convert_binary=True)
    # df = get_movielens1m(convert_binary=False)

    data = Data(seed=42)
    t0 = time()
    training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=0.5, train_precent=1)

    low_rank_cf_model = SimpleCF()

    best_hr=0
    best_ndcg = 0
    low_rank_cf_model.get_model(n_users, n_movies, n_latent_factors=128)
    t0 = time()
    mean_hr, mean_ndcg, time_eval = evaluate_model(low_rank_cf_model, test_set)
    print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.1f s]'
          % (mean_hr, mean_ndcg, time()-t0 ))
    for epoch in range(epochs):
        t1 = time()
        loss = low_rank_cf_model.fit_once(training_set, batch_size=128, verbose=0)
        t2 = time()
        mean_hr, mean_ndcg, time_eval = evaluate_model(low_rank_cf_model, test_set)
        t3 = time()
        print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
              % (epoch+1, t2 - t1, mean_hr, mean_ndcg, loss, t3 - t2))
        if mean_hr > best_hr and mean_ndcg > best_ndcg:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            if save_model:
                low_rank_cf_model.save_model()
    print('Total time: [%.1f s]' % (time()- t0))


if __name__ == '__main__':
    import os
    main()
