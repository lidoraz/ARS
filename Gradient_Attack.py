from DataLoader import *
from Data import Data, create_subset
from Evalute import evaluate_model
# from Models.SimpleCF import SimpleCF
from ga_attack_train_baseline import load_base_model
from Visualization.visualization import plot_interations, plot_interactions_neg
import tensorflow as tf
import os
import Constants
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np


def get_batches(training_set, batch_size):
    # training_mat = np.c_[np.column_stack(training_set), indexes]
    training_mat = np.column_stack(training_set)
    np.random.shuffle(training_mat)
    batches = np.array_split(training_mat, training_mat.shape[0] // batch_size)
    return batches

def create_attack_input(n_fake_users, n_users, n_movies, train_set, attack_rating_init_prob=0.5):
    """
    THERE ARE 1682 movies
    Creates the attack input that will be used to generate best fake ratings combination.
    It creates every combination of fake_user * item option and a initial rating prob.
    Then, it concatenetes the real training set in the end of each list of user, item, rating
    :param n_fake_users:
    :param n_users:
    :param n_movies:
    :param train_set:
    :param attack_rating_init_prob:
    :return: A tuple of (user, item, rating), list of attack users, list of attack_items, output_dim used as indexer, every entry smaller is related to fake user input.
    """
    # create an array with fake_users*items (note for +1 for inclusive of np.arange)
    cartesian_prodcut = np.array([[n_users + user - 1, item]
                                  for user in np.arange(1, n_fake_users + 1) for item in np.arange(0, n_movies)])
    attack_users = cartesian_prodcut[:, 0]
    attack_items = cartesian_prodcut[:, 1]
    attack_output_dim = n_fake_users * n_movies
    attack_ratings_prob = np.full((attack_output_dim,), attack_rating_init_prob)
    # # swap 1 to 0 and 1 to 0 from legitimate training set: #TODO test
    # real_rating = 1 - train_set[2]
    attack_input = [np.append(attack_users, train_set[0]),
                    np.append(attack_items, train_set[1]),
                    np.append(attack_ratings_prob, train_set[2])]
    real_training_rating_size = len(train_set[2])
    print('attack rating size=', attack_output_dim)
    print('real_training_rating_size=', real_training_rating_size)
    assert np.max(attack_users) == n_users + n_fake_users - 1
    assert len(np.unique(attack_users)) == n_fake_users
    assert len(attack_input[0]) == len(attack_input[1])
    assert len(attack_input[0]) == len(attack_input[2])

    # every index that is SMALL from output_dim is related to a fake user input
    return attack_input, attack_users, attack_items, attack_output_dim

def filter_ratings_by_treshold(attack_users, attack_items, attack_rating_prob, threshold):
    """

    :param attack_users:
    :param attack_items:
    :param attack_rating_prob:
    :param threshold:
    :return: attack_df_filterd, len(attack_df_filterd)
    """
    # for each user, we want to keep all positive ones, and keep excatly positive_ones* 4 zeros.
    import pandas as pd
    pd.options.mode.chained_assignment = None
    base_attack_df = pd.DataFrame({'u': attack_users, 'i': attack_items, 'r': attack_rating_prob})
    # maybe its better to take top ## instead of rint
    # fig = base_attack_df.r.hist(bins=10) # TODO: make plot
    # base_attack_df.apply(lambda x: x.r_prob > threshold, axis=1)
    if threshold <= 1:
        attack_df_filterd = base_attack_df[base_attack_df['r'] > threshold]  # apply on rows, keep only high prob ratings
    else:
        # attack_df_filterd = df.groupby('u')['r'].nlargest(threshold)
        attack_df_filterd = base_attack_df.sort_values('r', ascending=False).groupby('u').head(threshold)
    attack_df_filterd['r'] = 1
    print('Amount of total poison after filter:', len(attack_df_filterd))
    return attack_df_filterd, len(attack_df_filterd)

def mix_attack_df_with_training_set(mal_training_set, train_set_subset):
    from Data import concat_and_shuffle
    attack_benign_training_set = concat_and_shuffle(mal_training_set, train_set_subset)
    print(f'attack_benign_training_set={len(attack_benign_training_set[0])}, mal_training_set={len(mal_training_set[0])},'
          f' train_set_subset={len(train_set_subset[0])}')
    return attack_benign_training_set

def add_negative_samples(attack_df_filterd, n_movies, num_negatives):
    """
    get the attacks df and adds a negative sampling for each item, according to training policy.
    combines the df with percentage of training set
    :param attack_df_filterd:
    :param num_negatives:
    :return: list of (user, item, rating)
    """

    df = attack_df_filterd
    negative_items = {}
    users = list(sorted(df['u'].unique()))
    for idx, user in enumerate(users):
        negative_items[user] = np.setdiff1d(np.arange(n_movies), attack_df_filterd[attack_df_filterd.u == user].i.values)

    user_input, item_input, labels = [], [], []
    for index, row in df.iterrows():
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
    return mal_training_set


def generate_fake_ratings_random(model, n_fake_users, n_movies, attack_input):
    output_dim = n_fake_users * n_movies
    attack_rating_prob = np.random.rand(output_dim)
    return attack_rating_prob

def generate_fake_ratings_keras(model, n_fake_users, n_movies, attack_input):
    """
       performs gradient step in order to get which ratings should be changed to increase the loss
       :return:
       """
    import keras
    output_dim = n_fake_users * n_movies
    user_inp_p = model.inputs[0]
    item_inp_p = model.inputs[1]
    out_rating_p = model.outputs[0]

    sess = keras.backend.get_session()

    rating_p = tf.placeholder(tf.float32, [None, 1], name='ratings_prob')
    eps_p = tf.placeholder(tf.float32, [None, 1])  # each mal rating prob will take a eps_s * gradient step
    mask_p = tf.placeholder(tf.float32, [None, 1])  # used to mask out legitimate ratings

    # -THERA * (tf.reduce_sum(tf.rint(rating_input * mask_p)) / (n_fake_users * n_movies))
    lamda = 0
    reg_loss = lamda * (tf.reduce_sum(rating_p * mask_p) / n_fake_users)
    obj_loss = tf.keras.losses.binary_crossentropy(y_pred=out_rating_p, y_true=rating_p)
    # obj_loss = tf.keras.losses.mse(y_pred=out_rating_p, y_true=rating_p)
    combined_loss = reg_loss + obj_loss
    grad = tf.gradients(combined_loss, rating_p)  # take the gradient of the loss according to prediction # check
    rating_input_out = rating_p + (eps_p * grad * mask_p)  # perform a gradient step
    rating_input_out = tf.reshape(tf.clip_by_value(rating_input_out, 0, 1), (-1,))
    step = .001

    all_users, all_items, all_rating = attack_input
    indexes = np.arange(len(attack_input[0]))
    batch_size = 512
    batch_rating = None ; b_indexes = None; attack_rating_prob = None
    adv_l=None; obj_l=None; reg_l=None
    for i in range(5):
        # every epoch the values will get updated in attack input, under all_rating column

        for batch in get_batches(attack_input + [indexes], batch_size):
            b_users, b_items, b_ratings, b_indexes = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3].astype(
                'int')
            b_size = len(b_users)
            b_eps = np.full((b_size,), step)
            # mask the gradient such that benign user rating will not be changed
            b_mask = np.zeros((b_size,))
            b_mask[np.argwhere(b_indexes < output_dim)] = 1
            batch_rating, adv_l, obj_l, reg_l = sess.run([rating_input_out, combined_loss, obj_loss, reg_loss],
                                                         feed_dict={user_inp_p: b_users.reshape((-1, 1)),
                                                                    item_inp_p: b_items.reshape((-1, 1)),
                                                                    rating_p: b_ratings.reshape((-1, 1)),
                                                                    mask_p: b_mask.reshape((-1, 1)),
                                                                    eps_p: b_eps.reshape((-1, 1))})
            batch_rating = batch_rating.reshape((-1,))
            np.put(all_rating, b_indexes, batch_rating)

        attack_rating_prob = all_rating[np.argwhere(indexes < output_dim)][np.arange(output_dim)].reshape(-1, )
        # Extract attack rating from all rating by taking small index
        legit_rating = batch_rating[np.argwhere(b_indexes > output_dim)].reshape(-1, )
        assert len(np.unique(legit_rating)) == 2, 'legit rating must not change during gradient step'
        above_t = (attack_rating_prob > 0.5).sum()
        # if np.array_equal(previous_arr, attack_rating_prob > 0.5):
        #     print('same array...')
        # previous_arr = attack_rating_prob > 0.5
        below_t = (attack_rating_prob < 0.5).sum()
        print(i, np.round(attack_rating_prob[:10], 2), np.mean(attack_rating_prob), below_t, above_t, np.mean(adv_l), np.mean(obj_l),
              np.mean(reg_l))
    return attack_rating_prob

def train_attack_evalute_keras(model, attack_benign_training_set, test_set, batch_size, mal_epochs):
    # from ga_attack_multiprocess import load_base_model
    from Evalute import pert_train_evaluate_model
    # attack_benign_training_set = concat_and_shuffle(malicious_training_set, train_set)
    mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)
    print('evaluate_model:', mean_hr, mean_ndcg, time_eval)
    best_pert_model, best_pert_hr, best_pert_ndcg = pert_train_evaluate_model(model, attack_benign_training_set,
                                                                              test_set,
                                                                              batch_size=batch_size,
                                                                              epochs=mal_epochs,
                                                                              verbose=2)
    return 0, best_pert_hr, best_pert_ndcg


blackbox_poison_dir = "blackbox_poison"


def save_attack_benign_training_set(attack_benign_training_set, DATASET_NAME, test_type, train_frac, threshold):
    file_name = os.path.join(blackbox_poison_dir, f'ds{DATASET_NAME}_type{test_type}_t{train_frac}_t{threshold}.dump')
    pickle.dump(attack_benign_training_set, open(file_name, 'wb'))


def open_attack_benign_training_set(DATASET_NAME, test_type, train_frac, threshold):
    file_name = os.path.join(blackbox_poison_dir, f'ds{DATASET_NAME}_type{test_type}_t{train_frac}_t{threshold}.dump')
    attack_benign_training_set = pickle.load(open(file_name, 'rb'))
    return attack_benign_training_set

def run_experiment(exp_params, global_exp_params):
    tf.reset_default_graph()
    train_set_subset = exp_params['train_set_subset']
    n_fake_users = exp_params['n_fake_users']
    threshold = exp_params['threshold']
    test_set = exp_params['test_set']
    test_type = exp_params['test_type']
    train_frac = exp_params['train_frac']
    train_set = global_exp_params['train_set']
    DATASET_NAME = global_exp_params['DATASET_NAME']
    CONVERT_BINARY = global_exp_params['CONVERT_BINARY']
    num_negatives = global_exp_params['num_negatives']
    MAL_EPOCHS = global_exp_params['MAL_EPOCHS']
    n_users = global_exp_params['n_users']
    n_movies = global_exp_params['n_movies']
    plot = False
    model_name = global_exp_params['model_name']
    print('Started...')
    model, best_hr, best_ndcg = load_base_model(n_fake_users, DATASET_NAME, CONVERT_BINARY, model_name=model_name)
    #ATTACK
    attack_input, attack_users, attack_items, output_dim = create_attack_input(n_fake_users, n_users, n_movies, train_set)
    if test_type == 'random':
        attack_rating_prob = generate_fake_ratings_random(None, n_fake_users, n_movies, None)
    else:
        attack_rating_prob = generate_fake_ratings_keras(model, n_fake_users, n_movies, attack_input)
    attack_df_filtered, poison_amount = filter_ratings_by_treshold(attack_users, attack_items, attack_rating_prob, threshold)
    if plot:
        plot_interations(attack_df_filtered, DATASET_NAME, 2, 'Attack')
    # ATTACK EVALUATION
    mal_training_set = add_negative_samples(attack_df_filtered, n_movies, num_negatives)
    attack_benign_training_set = mix_attack_df_with_training_set(mal_training_set, train_set_subset)
    save_attack_benign_training_set(attack_benign_training_set, DATASET_NAME, test_type, train_frac, threshold)
    batch_size = 512
    epoch_loss, mean_hr, mean_ndcg = train_attack_evalute_keras(model, attack_benign_training_set, test_set, batch_size, MAL_EPOCHS)
    return mean_hr, mean_ndcg, poison_amount, best_hr - mean_hr



import pickle
import pandas as pd


def main():
    num_negatives = 4  # according to paper.
    MAL_EPOCHS = 3

    train_frac_list = [0.01]
    n_exp_users = 20
    n_exp_tresholds = 10
    # n_fake_user_list = [2, 4, 8 , 16, 32, 64, 128, 256]
    n_fake_user_list = [16]
    DATASET_NAME = 'movielens100k'
    # test_types = ['grad', 'random']
    test_types = ['random']
    model_name = 'simple_cf'
    plot = False
    CONVERT_BINARY = True
    # test_type = 'grad'
    # DATASET_NAME = 'movielens1m'
    # PREPROCESS
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)
    data = Data(seed=42)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=1)
    # tresholds_list = np.linspace(2, n_movies//10, n_exp_tresholds).astype(int)
    tresholds_list = [500] # cant be larger than n_movies

    if plot:
        plot_interactions_neg(train_set, DATASET_NAME, atleast=2, title='Full Training_set')

    for ii, train_frac in enumerate(train_frac_list):
        train_set_subset = create_subset(train_set, train_frac, DATASET_NAME, Constants.unique_subset_id)
        global_exp_params = {
            'train_set': train_set,
            'DATASET_NAME': DATASET_NAME,
            'CONVERT_BINARY': CONVERT_BINARY,
            'num_negatives': num_negatives,
            'MAL_EPOCHS': MAL_EPOCHS,
            'n_users': n_users,
            'n_movies': n_movies,
            'model_name': model_name
        }
        for test_type in test_types:
            rows = []
            for i, n_fake_users in enumerate(n_fake_user_list):
                for j, threshold in enumerate(tresholds_list):
                    exp_params = {
                        'train_set_subset': train_set_subset,
                        'n_fake_users': n_fake_users,
                        'threshold': threshold,
                        'test_set': test_set,
                        'test_type': test_type,
                        'train_frac': train_frac
                    }
                    mean_hr, mean_ndcg, poison_amount, delta_mean_hr = run_experiment(exp_params, global_exp_params)
                    print(f'Finished exp with: train_frac={train_frac:0.4f}, n_fake_users={n_fake_users}, threshold={threshold}, mean_hr= {mean_hr:0.2f} delta_mean_hr={delta_mean_hr:0.3f}')
                    rows.append([train_frac, n_fake_users, threshold, mean_hr, delta_mean_hr, mean_ndcg, poison_amount])
            results = pd.DataFrame(np.array(rows),
                                   columns=['train_frac', 'n_fake_users', 'threshold', 'mean_hr', 'delta_mean_hr',
                                            'mean_ndcg', 'poison_amount'])
            pickle.dump(results, open(f'out_results__{DATASET_NAME}_ALL_keras_{test_type}.pickle', 'wb'))

# pd.options.display.max_rows = 1000



# attack_params = {'n_fake_users': n_fake_users,
#                      'threshold': threshold,
#                      'theta': theta,
#                      'train_frac': train_frac,
#                      'testset_percentage': testset_percentage,
#     }

if __name__ == '__main__':
    main()