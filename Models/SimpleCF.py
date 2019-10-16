import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

checkpoint_cf_dir = "checkpoint_cf"  # "Directory name to save the checkpoints [checkpoint]")


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
    def __init__(self, df):
        self.df = df
        self.is_trained = False
        self.model = None
        self.RATING_MATRIX = None

    def load_model(self, model_path='{}/CF.json'.format(checkpoint_cf_dir),
                   weights_path='{}/CF_w.h5'.format(checkpoint_cf_dir)):

        from tensorflow.keras.models import model_from_json
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights_path)
        self.is_trained = True
        print('loaded model from:', model_path, 'weights_path:', weights_path)

    def model_preprocessing(self):
        users = self.df.user_id.unique()
        movies = self.df.movie_id.unique()

        self._userid2idx = {o: i for i, o in enumerate(users)}
        self._itemid2idx = {o: i for i, o in enumerate(movies)}
        self.df['user_id'] = self.df['user_id'].apply(lambda x: self._userid2idx[x])
        self.df['movie_id'] = self.df['movie_id'].apply(lambda x: self._itemid2idx[x])
        split = np.random.rand(len(self.df)) < 0.8
        train = self.df[split]
        valid = self.df[~split]

        self.n_movies = len(self.df['movie_id'].unique())
        self.n_users = len(self.df['user_id'].unique())
        print('n_users:', self.n_users, 'n_movies:', self.n_movies)
        print(train.shape, valid.shape)
        return train, valid, self.n_users, self.n_movies

    # includes model.compile with Adam optimizer, loss is MSE
    def set_model(self, n_users, n_movies, n_latent_factors=64):
        user_input = Input(shape=(1,), name='user_input', dtype='int64')
        user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)
        movie_input = Input(shape=(1,), name='movie_input', dtype='int64')
        movie_embedding = Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

        sim = dot([user_vec, movie_vec], name='Simalarity-Dot-Product', axes=1)
        model = Model([user_input, movie_input], sim)

        print(model.summary())
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        self.model = model
        return model

    def fit(self, train, valid, batch_size=128, epochs=25, verbose = 0 ):

        history = self.model.fit([train.user_id, train.movie_id], train.rating, batch_size=batch_size,
                                 epochs=epochs, validation_data=([valid.user_id, valid.movie_id], valid.rating),
                                 verbose=verbose)
        self.is_trained = True
        self.save_model()
        return 0

    def save_model(self):
        print('Saving model and weights at dir:', checkpoint_cf_dir)
        self.model.save('{}/CF.h5'.format(checkpoint_cf_dir), True, True, 'h5')
        self.model.save_weights('{}/CF_w.h5'.format(checkpoint_cf_dir))
        model_json = self.model.to_json()
        with open('{}/CF.json'.format(checkpoint_cf_dir), "w") as json_file:
            json_file.write(model_json)
        print('Saved model and weights successfully at dir:', checkpoint_cf_dir)

    def create_rating_matrix(self):
        user_mat = self.model.get_layer('user_embedding').get_weights()[0]
        item_mat = self.model.get_layer('movie_embedding').get_weights()[0]
        print('user_mat.shape', user_mat.shape)
        print('item_mat.shape', item_mat.shape)
        assert user_mat.shape[0] == self.n_users and item_mat.shape[0] == self.n_movies
        self.RATING_MATRIX = np.dot(user_mat, item_mat.T)

    def predict(self, users_items, batch_size, verbose):
        users = users_items[0]
        items = users_items[1]

        assert len(users) == len(items)
        if not self.is_trained:
            raise EnvironmentError('Cannot predict while model is not trained or init')
        if self.RATING_MATRIX is None:
            print('Creating RATING_MATRIX..')
            self.create_rating_matrix()
            print('Finished Creating RATING_MATRIX..')

        predictions = []
        for idx in range(len(users)):
            # We use the user_id -> index in order to get the right place in the matrix for corresponding user
            predictions.append(self.RATING_MATRIX[self._userid2idx[users[idx]], self._itemid2idx[items[idx]]])
        return np.array(predictions)


from DataLoader import *
from Data import Data
from Evalute import evaluate_model
ml1m = 'movielens1m'
ml100k = 'movielens100k'
def main():
    # An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
    # Results: Convert Binary state:
    # False: (0.1357370095440085, 0.06711691443559832)
    # true: (0.22693531283138918, 0.10654227847736084)

    convert_binary = False
    load_model = True

    epochs = 25
    # res epochs=15: mean_hr: 0.134 mean_ndcg: 0.055
    # res epoch=1: mean_hr: 0.443 mean_ndcg: 0.244 # TODO: WHAT??
    # res epoch=0 mean_hr: 0.1 mean_ndcg: 0.046
    dataset_name = ml100k
    # df, user_item_matrix, total_users, total_movies = get_movielens100k(convert_binary)
    df, user_item_matrix, total_users, total_movies = get_from_dataset_name(dataset_name, convert_binary)
    data = Data(df, user_item_matrix, total_users, total_movies, seed=42)

    test_set = data.create_testset()
    low_rank_cf_model = SimpleCF(df)
    train, valid, n_users, n_movies = low_rank_cf_model.model_preprocessing()

    if load_model:
        low_rank_cf_model.load_model()
    else:
        low_rank_cf_model.set_model(n_users, n_movies, n_latent_factors=64)
        history = low_rank_cf_model.fit(train, valid, batch_size=128, epochs=epochs, verbose=2)

    mean_hr, mean_ndcg = evaluate_model(low_rank_cf_model, test_set)

    # TODO: COMPARE HERE
    a = data.user_item_matrix
    b = low_rank_cf_model.RATING_MATRIX
    # TODO
    print('mean_hr:', np.round(mean_hr, 3), 'mean_ndcg:', np.round(mean_ndcg, 3))

if __name__ == '__main__':
    main()
