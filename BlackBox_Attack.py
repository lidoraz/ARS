from time import time
from DataLoader import *
from Data import Data
from Evalute import evaluate_model
from Models.SimpleCF import SimpleCF
# An example for running the model and evaluating using leave-1-out and top-k using hit ratio and NCDG metrics
convert_binary = True
load_model = False
save_model = False
testset_percentage = 0.2

epochs = 5
n_fake_users = 16 # TODO
print('Started...')

df = get_movielens100k(convert_binary=True)
# df = get_movielens1m(convert_binary=False)

data = Data(seed=42)
t0 = time()
training_set, test_set, n_users, n_movies = data.pre_processing(df, test_percent=0.5, train_precent=1)


best_hr = 0
best_ndcg = 0
batch_size = 512
n_users_w_mal = n_users + n_fake_users + 1
model = SimpleCF.get_model(n_users_w_mal, n_movies, n_latent_factors=2)
t0 = time()
mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set)
print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.1f s]'
      % (mean_hr, mean_ndcg, time() - t0))

user_input, item_input, labels = training_set
for epoch in range(epochs):
    t1 = time()
    loss = model.fit([user_input, item_input],  # input
                     labels,  # labels
                     batch_size=batch_size, verbose=0, shuffle=True).history['loss'][0]
    t2 = time()
    mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set)
    t3 = time()
    print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
          % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss, t3 - t2))
    if mean_hr > best_hr and mean_ndcg > best_ndcg:
        best_hr = mean_hr
        best_ndcg = mean_ndcg
        if save_model:
            model.save_model()

from Gradient_Attack import train_attack_evalute_keras, open_attack_benign_training_set
DATASET_NAME = 'movielens100k'
MAL_EPOCHS = 10

print('BLACK BOX')
attack_benign_training_set = open_attack_benign_training_set(DATASET_NAME, 'grad', 0.00, 168)
epoch_loss, mean_hr, mean_ndcg = train_attack_evalute_keras(model, attack_benign_training_set, test_set, batch_size, MAL_EPOCHS)

print('Total time: [%.1f s]' % (time() - t0))
