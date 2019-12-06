import keras
from keras.layers import Embedding, Input, Flatten

import tensorflow as tf

class Model:
    def __init__(self, num_users, num_items, mf_dim=8, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0):

        self.user_inp = tf.placeholder(dtype=tf.int32, shape=None, name='user_input')
        self.item_inp = tf.placeholder(dtype=tf.int32, shape=None, name='item_input')
        self.y_true = tf.placeholder(dtype=tf.float32, shape=None, name='y_true')


        mf_embedding_user = tf.Variable(tf.random_uniform([num_users, mf_dim]), name='mf_embedding_user')
        mf_embed_user = tf.nn.embedding_lookup(mf_embedding_user, self.user_inp)

        mf_embedding_item = tf.Variable(tf.random_uniform([num_items, mf_dim]), name='mf_embedding_item')
        mf_embed_item = tf.nn.embedding_lookup(mf_embedding_item, self.item_inp)

        mlp_embedding_user = tf.Variable(tf.random_uniform([num_users, int(layers[0] / 2)]), name='mlp_embedding_user')
        mlp_embed_user = tf.nn.embedding_lookup(mlp_embedding_user, self.user_inp)

        mlp_embedding_item = tf.Variable(tf.random_uniform([num_items, int(layers[0] / 2)]), name='mlp_embedding_item')
        mlp_embed_item = tf.nn.embedding_lookup(mlp_embedding_item, self.item_inp)

        # MF part
        mf_user_latent = tf.reshape(mf_embed_user, [-1])
        mf_item_latent = tf.reshape(mf_embed_item, [-1])
        mf_vector = tf.multiply(mf_user_latent, mf_item_latent)
        # MLP part
        mlp_user_latent = tf.reshape(mlp_embed_user, [-1])
        mlp_item_latent = tf.reshape(mlp_embed_item, [-1])
        mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], axis=-1)

        num_layer = len(layers)
        for idx in range(1, num_layer):
            mlp_vector = tf.layers.dense(mlp_vector, layers[idx], activation=tf.nn.relu, name="layer%d" % idx)
            # mlp_vector = layer(mlp_vector)

        predict_vector = tf.concat([mf_vector, mlp_vector], axis =-1)
        predict_vector = tf.expand_dims(predict_vector, axis=-1)
        self.prediction = tf.layers.dense(predict_vector,  1, name = "prediction")

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y_true, logits=self.prediction) # (Y_TRUE - PREDICTION)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train = self.optimizer.minimize(self.loss)

import numpy as np


import keras
from keras import initializers
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout

def run_model():
    SEED = 42
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                         seed=SEED),
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                         seed=SEED),
                                  embeddings_regularizer=l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                          seed=SEED),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                          seed=SEED),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    # mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    # mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    mlp_vector = keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), bias_regularizer=l2(reg_layers[idx]),
                      activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    # predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])

    # Final prediction layer
    # activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'
    prediction = Dense(1,name="prediction")(predict_vector)

    # model = Model(inputs=[user_input, item_input],
    #               outputs=[prediction])

    return user_input, item_input, prediction





def get_batch(training_set, batch_size ):
    training_mat = np.column_stack(training_set)
    # return np.vsplit(training_mat, training_mat.shape[0] // batch_size)
    batches = np.array_split(training_mat, training_mat.shape[0] // batch_size)
    return batches
    # print(len(batches))

from DataLoader import get_from_dataset_name
from Data import Data
with tf.Session() as sess:
    df = get_from_dataset_name('movielens100k', True)

    data = Data(seed=42)
    # training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=1.0)


    # num_users = n_users
    # num_items = n_movies
    num_users = 945
    num_items = 1096
    batch_size = 4
    user = np.random.randint(0,num_users,(4096,))
    item = np.random.randint(0,num_items,(4096,))
    rating = np.random.randint(0,2,(4096,))
    training_set = (user, item, rating)

    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0

    learning_rate = 0.001

    # model = Model(num_users = num_users,
    #               num_items = num_items,
    #               mf_dim = mf_dim,
    #               layers= layers,
    #               reg_layers=reg_layers,
    #               reg_mf=reg_mf)

    epochs = 10
    user_input, item_input, prediction = run_model()
    y_true = tf.placeholder(dtype=tf.float32, shape=None, name='y_true')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                        logits=prediction)  # (Y_TRUE - PREDICTION)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(loss)
    for e in range(epochs):

        batches = get_batch(training_set, batch_size= batch_size)

        sess.run(tf.global_variables_initializer())
        loses_in_epoch = []
        for b in batches:
            user = b[:,0].reshape(-1, 1)
            item = b[:,1].reshape(-1, 1)
            rating = b[:,2].reshape(-1, 1)
            # inp_user = None
            # item_inp = None
            _, preds, loss = sess.run([train, prediction, loss], {user_input: user,
                                                                                    item_input: item,
                                                                                    y_true: rating})
            # _, preds, loss = sess.run([model.train, model.prediction, model.loss], {model.user_inp: user,
            #                                                                  model.item_inp: item,
            #                                                                 model.y_true: rating})
            loses_in_epoch.append(np.mean(loss))
        print(np.mean(loses_in_epoch))
