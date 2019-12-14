import os
# from Constants import SEED
from NeuMF import get_model
from Evalute import baseline_train_evalute_model
from Data import *
import json

os.environ['RUN_MODE'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ml1m = 'movielens1m'
ml100k = 'movielens100k'

BASE_MODEL_DIR = 'base_models'

np.random.seed(SEED)


# Dataset Related
CONVERT_BINARY = True
DATASET_NAME = ml100k
# DATASET_NAME = ml1m
TEST_SET_PERCENTAGE = 1
BASE_MODEL_EPOCHS = 15  # will get the best model out of these n epochs.

VERBOSE = 1  # Verbose: 2 - print all in addition to iteration for each agent.


def main(n_fake_users=10):
    """
    this should be the best model according to the evalute process, in terms of HR and NDCG
    """
    df = get_from_dataset_name(DATASET_NAME, CONVERT_BINARY)
    data = Data(seed=SEED)
    train_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=TEST_SET_PERCENTAGE)
    n_users_w_mal = n_users + n_fake_users + 1

    # NeuMF Parameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    loss_func = 'binary_crossentropy'

    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={BASE_MODEL_EPOCHS}'
    model_path = f'{BASE_MODEL_DIR}/{params_name}.json'
    metrics_path = f'{BASE_MODEL_DIR}/{params_name}_metrics.json'
    weights_path = f'{BASE_MODEL_DIR}/{params_name}_w.h5'
    if not os.path.exists(model_path):
        print(model_path, 'does not exists.. creating a baseline model')
        print('REGULAR PHASE')
        model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
        from keras.optimizers import Adam
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
        print('get_model done')
        model_base, best_hr, best_ndcg = baseline_train_evalute_model(model, train_set, test_set, batch_size=batch_size,
                                                                      epochs=BASE_MODEL_EPOCHS)
        print('baseline_train_evalute_model done best_hr = {}'.format(best_hr))
        model_json = model_base.to_json()
        model_base.save_weights(weights_path)
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        with open(metrics_path, 'w') as metrics_file:
            json.dump({'best_hr': best_hr, 'best_ndcg': best_ndcg}, metrics_file)
        print('Saved model and weights at dir:', BASE_MODEL_DIR)
    else:
        print('model exists!')

import fire

if __name__ == '__main__':
    fire.Fire(main)
