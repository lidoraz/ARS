import Constants
import os
from Data import get_train_test_set
import json

from Evalute import baseline_train_evalute_model


def get_weights(n_fake_users, DATASET_NAME):
    import json
    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={Constants.BASE_MODEL_EPOCHS}'
    model_path = f'{Constants.BASE_MODEL_DIR}/{params_name}.json'
    metrics_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_metrics.json'
    weights_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_w.h5'

    assert os.path.exists(model_path), f'Model does not exists at: {model_path}'
    with open(metrics_path, 'r') as metrics_file:
        # model = load_base_model(n_fake_users)
        metrics = json.load(metrics_file)
        best_hr = metrics['best_hr']
        best_ndcg = metrics['best_ndcg']
    return weights_path, best_hr, best_ndcg


def load_base_model(n_fake_users, DATASET_NAME, convert_binary):
    import json
    from keras.models import model_from_json
    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={Constants.BASE_MODEL_EPOCHS}'
    model_path = f'{Constants.BASE_MODEL_DIR}/{params_name}.json'
    metrics_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_metrics.json'
    weights_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_w.h5'
    if not os.path.exists(model_path):
        print('Keras model does not exists, training....')
        train_base_model(n_fake_users, DATASET_NAME, convert_binary)

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    with open(metrics_path, 'r') as metrics_file:
        # model = load_base_model(n_fake_users)
        metrics = json.load(metrics_file)
        best_hr = metrics['best_hr']
        best_ndcg = metrics['best_ndcg']
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    return model, best_hr, best_ndcg

def train_base_model(n_fake_users, DATASET_NAME, CONVERT_BINARY):
    """
    this should be the best model according to the evalute process, in terms of HR and NDCG
    """
    train_set, test_set, n_users, n_movies = get_train_test_set(DATASET_NAME, CONVERT_BINARY, Constants.SEED)
    n_users_w_mal = n_users + n_fake_users + 1

    # NeuMF Parameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    learning_rate = 0.001
    batch_size = 512
    loss_func = 'binary_crossentropy'

    params_name = f'NeuMF_{DATASET_NAME}_u={n_fake_users}_e={Constants.BASE_MODEL_EPOCHS}'
    model_path = f'{Constants.BASE_MODEL_DIR}/{params_name}.json'
    metrics_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_metrics.json'
    weights_path = f'{Constants.BASE_MODEL_DIR}/{params_name}_w.h5'
    if not os.path.exists(model_path):
        print(model_path, 'does not exists.. creating a baseline model')
        print('REGULAR PHASE')
        from Models.NeuMF import get_model
        model = get_model(n_users_w_mal, n_movies, mf_dim, layers, reg_layers, reg_mf)
        from keras.optimizers import Adam
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
        print('get_model done')
        model_base, best_hr, best_ndcg = baseline_train_evalute_model(model, train_set, test_set, batch_size=batch_size,
                                                                      epochs=Constants.BASE_MODEL_EPOCHS)
        print('baseline_train_evalute_model done best_hr = {}'.format(best_hr))
        model_json = model_base.to_json()
        model_base.save_weights(weights_path)
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        with open(metrics_path, 'w') as metrics_file:
            json.dump({'best_hr': best_hr, 'best_ndcg': best_ndcg}, metrics_file)
        print('Saved model and weights at dir:', Constants.BASE_MODEL_DIR)
    else:
        print('model exists!')

import fire

if __name__ == '__main__':
    fire.Fire(train_base_model)
