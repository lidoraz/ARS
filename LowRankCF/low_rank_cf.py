import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

checkpoint_cf_dir = "checkpoint_cf"  # "Directory name to save the checkpoints [checkpoint]")


def get_pivot_table(df):
    util_df = pd.pivot_table(data=df, values='rating', index='user_id', columns='movie_id').fillna(0)
    return util_df
def get_data():
    DATA_PATH = 'E:/DEEP_LEARNING/DATA_SETS/ml-100k/u.data'

    df = pd.read_csv(DATA_PATH, delimiter='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp']) \
        .drop(columns=['timestamp'])

    index = list(df['user_id'].unique())
    columns = list(df['movie_id'].unique())
    index = sorted(index)
    columns = sorted(columns)

    # util_df = pd.pivot_table(data=df, values='rating', index='user_id', columns='movie_id').fillna(0)

    return df


def model_preprocessing(df):
    users = df.user_id.unique()
    movies = df.movie_id.unique()

    userid2idx = {o: i for i, o in enumerate(users)}
    movieid2idx = {o: i for i, o in enumerate(movies)}
    df['user_id'] = df['user_id'].apply(lambda x: userid2idx[x])
    df['movie_id'] = df['movie_id'].apply(lambda x: movieid2idx[x])
    split = np.random.rand(len(df)) < 0.8
    train = df[split]
    valid = df[~split]
    print(train.shape, valid.shape)
    n_movies = len(df['movie_id'].unique())
    n_users = len(df['user_id'].unique())
    return train, valid, n_users, n_movies


def get_model(n_users, n_movies, n_latent_factors=64):
    user_input = Input(shape=(1,), name='user_input', dtype='int64')
    user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
    user_vec = Flatten(name='FlattenUsers')(user_embedding)
    movie_input = Input(shape=(1,), name='movie_input', dtype='int64')
    movie_embedding = Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
    movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

    sim = dot([user_vec, movie_vec], name='Simalarity-Dot-Product', axes=1)
    model = Model([user_input, movie_input], sim)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model


def fit_model(model, train, valid, batch_size = 128, epochs = 10):

    history = model.fit([train.user_id, train.movie_id], train.rating, batch_size=batch_size,
                        epochs=epochs, validation_data=([valid.user_id, valid.movie_id], valid.rating),
                        verbose=1)
    return history


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


def get_user_mat(model, user_id):
    user_mat = model.get_layer('user_embedding').get_weights()[0]
    return user_mat[user_id]


def get_user_item_embeddings(model):
    user_mat = model.get_layer('user_embedding').get_weights()[0]
    item_mat = model.get_layer('movie_embedding').get_weights()[0]
    return user_mat, item_mat


def main():
    training = False

    df = get_data()
    if training == True:
        df = get_data()
        train, valid, n_users, n_movies = model_preprocessing(df)
        model = get_model(n_users, n_movies, n_latent_factors=64)
        history = fit_model(model, train, valid, epochs= 100)
        plot_loss(history)
        model.save('{}/CF.h5'.format(checkpoint_cf_dir), True, True, 'h5')
        get_user_mat(model, 215)
        model.save_weights('{}/CF_w.h5'.format(checkpoint_cf_dir))
        model_json = model.to_json()
        with open('{}/CF.json'.format(checkpoint_cf_dir), "w") as json_file:
            json_file.write(model_json)
    else:
        from tensorflow.keras.models import model_from_json
        with open('{}/CF.json'.format(checkpoint_cf_dir), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights('{}/CF_w.h5'.format(checkpoint_cf_dir))

        user_mat, item_mat = get_user_item_embeddings(model)
        print(user_mat.shape)
        print(item_mat.shape)

        RATING_MATRIX = np.dot(user_mat, item_mat.T)
        pivot = get_pivot_table(df)
        RATING_MATRIX.shape


if __name__ == '__main__':
    main()
