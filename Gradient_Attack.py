from DataLoader import *
from Data import Data
from Evalute import evaluate_model
# from Models.SimpleCF import SimpleCF
# from NeuMF import get_model

import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, num_users, num_items, mf_dim=8, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0):
        with tf.variable_scope('NeuCF_network'):
            self.user_inp = tf.placeholder(dtype=tf.int32, shape=[None], name='user_input')
            self.item_inp = tf.placeholder(dtype=tf.int32, shape=[None], name='item_input')
            self.y_true = tf.placeholder(dtype=tf.float32, shape=[None], name='y_true')

            normal_init = tf.random_normal_initializer(mean=0, stddev=0.01)
            pred_init = tf.keras.initializers.lecun_uniform(seed=None)
            # TODO: Add regulaizer

            mf_embedding_user = tf.Variable(normal_init([num_users, mf_dim]), name='mf_embedding_user')
            mf_embed_user = tf.nn.embedding_lookup(mf_embedding_user, self.user_inp)

            mf_embedding_item = tf.Variable(normal_init([num_items, mf_dim]), name='mf_embedding_item')
            mf_embed_item = tf.nn.embedding_lookup(mf_embedding_item, self.item_inp)

            mlp_embedding_user = tf.Variable(normal_init([num_users, int(layers[0] / 2)]), name='mlp_embedding_user')
            mlp_embed_user = tf.nn.embedding_lookup(mlp_embedding_user, self.user_inp)

            mlp_embedding_item = tf.Variable(normal_init([num_items, int(layers[0] / 2)]), name='mlp_embedding_item')
            mlp_embed_item = tf.nn.embedding_lookup(mlp_embedding_item, self.item_inp)

            # MF part
            mf_user_latent = tf.reshape(mf_embed_user, [-1, mf_embed_user.shape[-1]])
            mf_item_latent = tf.reshape(mf_embed_item, [-1, mf_embed_item.shape[-1]])
            mf_vector = tf.multiply(mf_user_latent, mf_item_latent)
            # MLP part
            mlp_user_latent = tf.reshape(mlp_embed_user, [-1, mlp_embed_user.shape[-1]])
            mlp_item_latent = tf.reshape(mlp_embed_item, [-1, mlp_embed_item.shape[-1]])
            mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], axis=-1)
            num_layer = len(layers)
            for idx in range(1, num_layer):
                mlp_vector = tf.layers.dense(mlp_vector, layers[idx], activation=tf.nn.relu, name="layer{}".format(idx))

            self.predict_vector = tf.concat([mf_vector, mlp_vector], axis=-1)

            prediction = tf.layers.dense(inputs=self.predict_vector,
                                         kernel_initializer=pred_init,
                                         bias_initializer=pred_init,
                                         units=1, activation='sigmoid')
            self.prediction = tf.reshape(prediction, [-1], name='prediction')
            self.loss = tf.keras.losses.binary_crossentropy(y_pred=self.prediction, y_true=self.y_true)
            #             self.loss = tf.reduce_mean((self.y_true*tf.log(self.prediction)) + ((1-self.y_true)*tf.log(1-self.prediction))) # (Y_TRUE - PREDICTION)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = self.optimizer.minimize(self.loss)

class TFPredictWrapper:
    def __init__(self, model, sess):
        self.model = model
        self.sess = sess

    def predict(self, user_item, verbose):
        [u, i] = user_item
        model = self.model
        preds = self.sess.run(model.prediction,
                         feed_dict={model.user_inp: u,
                                    model.item_inp: i})

        return preds

def get_model(n_fake_users, n_users, n_movies):
    num_users = n_fake_users + n_users
    num_items = n_movies
    batch_size = 512

    # model params:
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0

    learning_rate = 0.001
    model = Model(num_users=num_users,
                  num_items=num_items,
                  mf_dim=mf_dim,
                  layers=layers,
                  reg_layers=reg_layers,
                  reg_mf=reg_mf)
    print(f'model with: n_fake_users={n_fake_users}, n_users={n_users}, n_movies={n_movies} created')
    return model

def get_batches(training_set, batch_size):
    training_mat = np.column_stack(training_set)
    np.random.shuffle(training_mat)
    # return np.vsplit(training_mat, training_mat.shape[0] // batch_size)
    batches = np.array_split(training_mat, training_mat.shape[0] // batch_size)
    return batches

def train_model(N_FAKE_USERS, n_users, n_movies, train_set, test_set):
    import os
    model_path = f"tf_neumf_models/model{N_FAKE_USERS}.ckpt"
    if os.path.exists(model_path + '.index'):
        print("Model already exists at:", model_path)
        return
    epochs = 10
    batch_size = 512
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = get_model(n_fake_users=N_FAKE_USERS, n_users=n_users, n_movies=n_movies)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        model_p = TFPredictWrapper(model, sess)
        # training and validation
        for e in range(epochs):
            batches = get_batches(train_set, batch_size=batch_size)
            loses_in_epoch = []
            for b in batches:
                u = b[:, 0]
                i = b[:, 1]
                r = b[:, 2]
                _, preds, loss = sess.run([model.train_op, model.prediction, model.loss],
                                          feed_dict={model.user_inp: u,
                                                     model.item_inp: i,
                                                     model.y_true: r})
                loses_in_epoch.append(np.mean(loss))
            epoch_loss = np.mean(loses_in_epoch)

            mean_hr, mean_ndcg, time_eval = evaluate_model(model_p, test_set, verbose=0)
            print('e={} hr={:.3f} ndcg={:.3f} time={:.3f} train_loss={:.3f} '.format(e + 1, mean_hr, mean_ndcg,
                                                                                     time_eval,
                                                                                     epoch_loss))
        save_path = saver.save(sess, model_path)

def predict(n_fake_users, n_users, n_movies, test_set):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = get_model(n_fake_users=n_fake_users, n_users=n_users, n_movies=n_movies)
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        model_p = TFPredictWrapper(model, sess)
        mean_hr, mean_ndcg, time_eval = evaluate_model(model_p, test_set, verbose=0)
        print('hr={:.3f} ndcg={:.3f} time={:.3f}'.format(mean_hr, mean_ndcg, time_eval))

def create_attack_input(n_fake_users, n_users, n_movies, train_set):
    """
    This create a tuple of cartesian product of fake_users * n_movies
    Which later will be fed to the algorithm to generate fake ratings probabilities
    :param n_fake_users:
    :param n_users:
    :param n_movies:
    :param train_set:
    :return:
    """
    output_dim = (n_fake_users - 1) * (n_movies - 1)
    cartesian_prodcut = np.array([[n_users + user - 1, item]
                                  for user in np.arange(1, n_fake_users) for item in np.arange(1, n_movies)])
    attack_users = cartesian_prodcut[:, 0]
    attack_items = cartesian_prodcut[:, 1]
    attack_user_item_cartesian = np.array([np.append(attack_users, train_set[0]),
                                 np.append(attack_items, train_set[1])])
    real_training_rating_size = len(train_set[2])
    print('attack rating size=', output_dim)
    print('real_training_rating_size=', real_training_rating_size)
    return attack_user_item_cartesian, attack_users, attack_items

def filter_ratings_by_treshold(attack_users, attack_items, attack_rating_prob, threshold):
    # for each user, we want to keep all positive ones, and keep excatly positive_ones* 4 zeros.
    import pandas as pd

    base_attack_df = pd.DataFrame({'u': attack_users, 'i': attack_items, 'r_prob': attack_rating_prob})
    # maybe its better to take top ## instead of rint
    # fig = base_attack_df.r.hist(bins=10) # TODO: make plot
    attack_df_filterd = base_attack_df[base_attack_df.apply(lambda x: x.r_prob > threshold, axis=1)]  # apply on rows, keep only high prob ratings
    attack_df_filterd['r'] = attack_df_filterd['r_prob'].apply(lambda x: 1)  # |.astype(int)
    print('Amount of total poison after filter:', len(attack_df_filterd))
    return attack_df_filterd

def mix_attack_df_with_training_set(mal_training_set, train_set, TRAIN_FRAC=0.1):
    from Data import create_subset, concat_and_shuffle
    print('train_set', len(train_set[0]))
    train_set_subset = create_subset(train_set, train_frac=TRAIN_FRAC)
    attack_benign_training_set = concat_and_shuffle(mal_training_set, train_set_subset)
    print(len(mal_training_set[0]))
    len(attack_benign_training_set[0])
    return attack_benign_training_set

def add_negative_samples(attack_df_filterd, num_negatives):
    """
    get the attacks df and adds a negative sampling for each item, according to training policy.
    combines the df with percentage of training set
    :param attack_df_filterd:
    :param num_negatives:
    :return: list of (user, item, rating)
    """
    from tqdm import tqdm

    df = attack_df_filterd
    negative_items = {}
    users = list(sorted(df['u'].unique()))
    for idx, user in tqdm(enumerate(users), total=len(users)):
        negative_items[user] = df[df['u'] != user]['i'].unique()

    df = df.groupby('u').apply(lambda s: s.sample(frac=1))
    user_input, item_input, labels = [], [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        user = row['u']
        user_input.append(user)
        item_input.append(row['i'])
        labels.append(row['r'])
        negative_input_items = np.random.choice(negative_items[user], num_negatives)
        for neg_item in negative_input_items:
            user_input.append(user)
            item_input.append(neg_item)
            labels.append(0)

    mal_training_set = (np.array(user_input), np.array(item_input), np.array(labels))
    return mal_training_set  # this already has the inputs and negative items

def generate_fake_ratings(n_fake_users, n_users, n_movies, train_set, attack_train_set):
    """
    performs gradient step in order to get which ratings should be changed to increase the loss
    :return:
    """
    model_path = f"tf_neumf_models/model{n_fake_users}.ckpt"
    output_dim = (n_fake_users - 1) * (n_movies - 1)
    real_training_rating_size = len(train_set[0])
    tf.reset_default_graph()
    mal_real_shape = output_dim + real_training_rating_size
    rating_input = tf.placeholder(tf.float32, [mal_real_shape], name='ratings_prob')
    model = get_model(n_fake_users=n_fake_users, n_users=n_users, n_movies=n_movies)
    print('rating_input.shape:', rating_input.shape)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        eps_p = tf.placeholder(tf.float32, mal_real_shape)
        mask_p = tf.placeholder(tf.float32, mal_real_shape)

        new_loss = tf.keras.losses.binary_crossentropy(y_pred=model.prediction, y_true=rating_input)
        grad = tf.gradients(new_loss, rating_input)  # take the gradient of the loss according to prediction # check
        rating_input_out = tf.stop_gradient(rating_input + (eps_p * grad * mask_p))  # perform a gradient step
        rating_input_out = tf.reshape(tf.clip_by_value(rating_input_out, 0, 1), (-1,))
        step = 1000
        eps = np.full((mal_real_shape,), step)
        # try to emulate a session with all data - training_set + adver, then the gradient will adjust adver rating params
        #     new_labels = np.full((output_dim,), 0.01) # initiate new ratings at 0.5
        new_labels_mal_part = np.full((output_dim,), 0.5)
        new_labels = np.concatenate([new_labels_mal_part, train_set[2]])
        all_users = attack_train_set[0]
        all_items = attack_train_set[1]
        indexes = np.arange(len(all_users))
        # new_labels.shape
        for i in range(20):
            ## todo: There is an issue in creating a shuffle while keeping track in the order of the mal data

            #         print(i)
            #         p = np.random.permutation(len(new_labels))
            #         all_users = all_users[p]
            #         all_items = all_items[p]
            #         new_labels = new_labels[p]
            #         new_labels_before_change = new_labels.copy()
            mask = np.zeros((mal_real_shape,))
            #         indexes = indexes[p] # this will help get the mal ratings in order as we started
            mask[np.argwhere(indexes < output_dim)] = 1

            new_labels, adv_l = sess.run([rating_input_out, new_loss],
                                         feed_dict={model.user_inp: all_users,
                                                    model.item_inp: all_items,
                                                    rating_input: new_labels,
                                                    mask_p: mask,
                                                    eps_p: eps})
            new_labels = new_labels.reshape((-1,))
            #         new_labels[np.argwhere(indexes >= output_dim)] = new_labels_before_change[np.argwhere(indexes >= output_dim)]
            attack_rating_prob = new_labels[np.argwhere(indexes < output_dim)][np.arange(output_dim)].reshape(-1, )
            ## Safety Check such that legit ratings did not change
            legit_rating = new_labels[np.argwhere(indexes > output_dim)].reshape(-1, )
            assert len(np.unique(legit_rating)) == 2, 'legit rating must not change during gradient step'
            print(i, np.round(attack_rating_prob[:10], 2), np.mean(attack_rating_prob))
        return attack_rating_prob


def train_attack_evalute(N_FAKE_USERS, attack_benign_training_set, test_set, mal_epochs, n_users, n_movies):
    # evaluate the model using the fake predictions
    tf.reset_default_graph()
    model = get_model(n_fake_users=N_FAKE_USERS, n_users=n_users, n_movies=n_movies)
    with tf.Session() as sess:
        model_p = TFPredictWrapper(model, sess)
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        for e in range(mal_epochs):
            u = attack_benign_training_set[0]
            i = attack_benign_training_set[1]
            r = attack_benign_training_set[2]
            _, preds, epoch_loss = sess.run([model.train_op, model.prediction, model.loss],
                                      feed_dict={model.user_inp: u,
                                                 model.item_inp: i,
                                                 model.y_true: r})

            mean_hr, mean_ndcg, time_eval = evaluate_model(model_p, test_set, verbose=0)
            print('e={} loss={:.3f} hr={:.3f} ndcg={:.3f} time[{:.3f}]s'.format(e, epoch_loss, mean_hr, mean_ndcg, time_eval))
    return epoch_loss, mean_hr, mean_ndcg

# def train_attack_evalute_batches(N_FAKE_USERS, attack_benign_training_set, test_set, batch_size, mal_epochs, n_users,
#                                  n_movies):
#     # evaluate the model using the fake predictions
#     tf.reset_default_graph()
#     model = get_model(n_fake_users=N_FAKE_USERS, n_users=n_users, n_movies=n_movies)
#
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         saver.restore(sess, "model.ckpt")
#         model_p = TFPredictWrapper(model, sess)
#         for e in range(mal_epochs):
#             batches = get_batches(attack_benign_training_set, batch_size=batch_size)
#             loses_in_epoch = []
#             for b in batches:
#                 u = b[:, 0]
#                 i = b[:, 1]
#                 r = b[:, 2]
#                 _, preds, loss = sess.run([model.train_op, model.prediction, model.loss],
#                                           feed_dict={model.user_inp: u,
#                                                      model.item_inp: i,
#                                                      model.y_true: r})
#                 loses_in_epoch.append(np.mean(loss))
#             epoch_loss = np.mean(loses_in_epoch)
#             mean_hr, mean_ndcg, time_eval = evaluate_model(model_p, test_set, verbose=0)
#             print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(epoch_loss, mean_hr, mean_ndcg, time_eval))
#     return epoch_loss, mean_hr, mean_ndcg


def run_expriment(n_fake_users, threshold) -> float:

    convert_binary = True
    load_model = False
    testset_percentage = 0.2
    num_negatives = 4 # according to paper..
    mal_epochs = 5
    TRAIN_FRAC = 0.1
    print('Started...')
    DATASET_NAME = 'movielens100k'
    # DATASET_NAME = 'movielens1m'
    # PREPROCESS
    df = get_from_dataset_name(DATASET_NAME, True)
    data = Data(seed=42)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=1)
    # TRAIN
    train_model(n_fake_users, n_users, n_movies, train_set, test_set) # creates a model and saves it
    predict(n_fake_users, n_users, n_movies, test_set)
    #ATTACK
    attack_user_item_cartesian, attack_users, attack_items = create_attack_input(n_fake_users, n_users, n_movies, train_set)
    attack_rating_prob = generate_fake_ratings(n_fake_users, n_users, n_movies, train_set, attack_user_item_cartesian)
    attack_df_filtered = filter_ratings_by_treshold(attack_users, attack_items, attack_rating_prob, threshold)
    # ATTACK EVALUATION
    mal_training_set = add_negative_samples(attack_df_filtered, num_negatives)
    attack_benign_training_set = mix_attack_df_with_training_set(mal_training_set, train_set, TRAIN_FRAC=TRAIN_FRAC)
    epoch_loss, mean_hr, mean_ndcg = train_attack_evalute(n_fake_users, attack_benign_training_set, test_set, mal_epochs, n_users, n_movies)

    return mean_hr


run_expriment(8, 0.6)



