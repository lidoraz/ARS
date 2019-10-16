import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

checkpoint_cf_dir = "checkpoint_cf"  # "Directory name to save the checkpoints [checkpoint]")

class SimpleCF():
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

    def model_preprocessing(self):
        users = self.df.user_id.unique()
        movies = self.df.movie_id.unique()
    
        userid2idx = {o: i for i, o in enumerate(users)}
        movieid2idx = {o: i for i, o in enumerate(movies)}
        self.df['user_id'] = self.df['user_id'].apply(lambda x: userid2idx[x])
        self.df['movie_id'] = self.df['movie_id'].apply(lambda x: movieid2idx[x])
        split = np.random.rand(len(self.df)) < 0.8
        train = self.df[split]
        valid = self.df[~split]
        print(train.shape, valid.shape)
        n_movies = len(self.df['movie_id'].unique())
        n_users = len(self.df['user_id'].unique())
        return train, valid, n_users, n_movies

    # includes model.compile with Adam optimizer, loss is MSE
    def get_model(self, n_users, n_movies, n_latent_factors=64):
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

    def fit(self, train, valid, batch_size = 128, epochs = 25):

        history = self.model.fit([train.user_id, train.movie_id], train.rating, batch_size=batch_size,
                            epochs=epochs, validation_data=([valid.user_id, valid.movie_id], valid.rating),
                            verbose=1)
        self.is_trained = True
        self.save_model()

        return history

    def save_model(self):
        print('Saving model and weights at dir:', checkpoint_cf_dir)
        self.model.save('{}/CF.h5'.format(checkpoint_cf_dir), True, True, 'h5')
        self.model.save_weights('{}/CF_w.h5'.format(checkpoint_cf_dir))
        model_json = self.model.to_json()
        with open('{}/CF.json'.format(checkpoint_cf_dir), "w") as json_file:
            json_file.write(model_json)
        print('Saved model and weights successfully at dir:', checkpoint_cf_dir)



    def plot_loss(self, history):
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


# def get_user_mat(model, user_id):
#     user_mat = model.get_layer('user_embedding').get_weights()[0]
#     return user_mat[user_id]

    def create_rating_matrix(self):
        user_mat = self.model.get_layer('user_embedding').get_weights()[0]
        item_mat = self.model.get_layer('movie_embedding').get_weights()[0]
        print('user_mat.shape', user_mat.shape)
        print('item_mat.shape', item_mat.shape)
        RATING_MATRIX = np.dot(user_mat, item_mat.T)
        self.RATING_MATRIX = RATING_MATRIX

    def predict(self, users, items):
        assert len(users) == len(items)
        if not self.is_trained:
            print('Cannot predict while model is not trained or init')
            return -1
        if self.RATING_MATRIX is None:
            print('Creating RATING_MATRIX..')
            self.create_rating_matrix()
            print('Finished Creating RATING_MATRIX..')

        predictions = []
        for idx in range(len(users)):
            predictions.append(self.RATING_MATRIX[users[idx], items[idx]])
        return np.array(predictions)

from DataLoader import get_movielens100k

def main():
    df, user_item_matrix, total_users, total_movies = get_movielens100k(convert_binary=False)
    low_rank_cf_model = SimpleCF(df)
    low_rank_cf_model.load_model() # path is already mentioned
    preds = low_rank_cf_model.predict([1 ,1, 1], [1 ,2, 3])
    print(preds)

if __name__ == '__main__':
    main()
