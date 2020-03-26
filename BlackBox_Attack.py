from time import time
from DataLoader import *
from Data import Data
from Evalute import evaluate_model
from Models.SimpleCF import SimpleCF
# An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
from ga_attack_train_baseline import load_base_model

convert_binary = True
load_model = False
save_model = False
testset_percentage = 0.2

epochs = 5
n_fake_users = 16 # TODO
print('Started...')

df = get_movielens100k(convert_binary=True)

data = Data(seed=42)
t0 = time()
training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=0.5, train_precent=1)

DATASET_NAME = 'movielens100k'
CONVERT_BINARY = True
model_name = 'simple_cf'

model, best_hr, best_ndcg = load_base_model(n_fake_users, DATASET_NAME, CONVERT_BINARY, model_name=model_name)

from Gradient_Attack import train_attack_evalute_keras, open_attack_benign_training_set
MAL_EPOCHS = 5

print('BLACK BOX')
batch_size = 512
attack_benign_training_set = open_attack_benign_training_set(DATASET_NAME, 'grad', 0.01, 168)
epoch_loss, mean_hr, mean_ndcg = train_attack_evalute_keras(model, attack_benign_training_set, test_set, batch_size, MAL_EPOCHS)

delta_hr = best_hr - mean_hr
print(f'Total damage: {delta_hr:0.4f} Total time: [{time() - t0:.1f} s]')
