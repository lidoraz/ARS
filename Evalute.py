"""
Based on Apr 15, 2016 Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16
Evaluate the performance of Top-k recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
"""
import math
import numpy as np
import gc
"""
Evaluation for MovieLens datasets
Metrics are Hit Ratio and NDCG, based on 10 Most recommended items.
*NOTE: These two metrics focus on RECOMMENDATION tools
Please note that batch size is 100 for the model.

    Evaluate the performance (Hit_Ratio, NDCG) of top-k recommendation
    Return: score of each test rating.
"""

from time import time


from Constants import SEED
np.random.seed(SEED)


def baseline_train_evalute_model(model, train_set, test_set, batch_size=512, epochs=5):
    from keras.models import clone_model
    best_hr = 0
    best_ndcg = 0
    best_epoch = 0
    models = []
    for epoch in range(epochs):
        t1 = time()
        (user_input, item_input, labels) = train_set
        loss = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, verbose=0, shuffle=True)
        t2 = time()
        mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)
        print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
              % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss.history['loss'][0], time_eval))
        model_copy = clone_model(model)
        model_copy.set_weights(model.get_weights())
        models.append(model_copy)
        if mean_hr > best_hr:
        # if mean_hr > best_hr or mean_ndcg > best_ndcg and epoch > 0:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            best_epoch = epoch

    return models[best_epoch], best_hr, best_ndcg


def pert_train_evaluate_model(model, train_set, test_set, batch_size=512, epochs=5, verbose= 0):
    best_hr = 1
    best_ndcg = 1
    best_epoch = 0
    user_input, item_input, labels = train_set
    # mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=2)
    for epoch in range(epochs):
        t1 = time()
        loss = model.fit([user_input, item_input],  # input
                         labels,  # labels
                         batch_size=batch_size, verbose=0, shuffle=True)
        t2 = time()
        mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)

        if verbose > 1:
            print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
                  % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss.history['loss'][0], time_eval))
        # TODO: CHANGED HERE FOR ATLEAST 2 EPOCHS, takeing the first epoch does not change because learning rate is small
        # take here worst hr value.
        if mean_hr < best_hr:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            best_epoch = epoch
    gc.collect()

    return best_epoch, best_hr, best_ndcg


def evaluate_model(model, test_set, k=10, verbose=1):
    t0 = time()
    user_list = list(test_set.keys())
    negatives_list = list(test_set.values())
    hits, ndcgs = [], []
    for idx in range(len(user_list)):     # Single thread
        (hr, ndcg) = eval_one_rating(model, idx, user_list, negatives_list, k)
        hits.append(hr)
        ndcgs.append(ndcg)
    mean_hr, mean_ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    time_eval = time() - t0
    if verbose > 1:
        print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.2f s]' % (mean_hr, mean_ndcg, time_eval))
    return mean_hr, mean_ndcg, time_eval


def eval_one_rating(model, idx, user_list, negatives_list, k):
    user = user_list[idx]

    gtItem = negatives_list[idx][-1]

    user_arr = np.full(len(negatives_list[idx]), user, dtype='int32')  # creates an array of size len(items) with values of u)
    items = np.array(negatives_list[idx])
    predictions = model.predict([user_arr, items],
                                 verbose=0)  # given user i and set of items, return probability for each of #batch_size items
    item_locations = (-predictions.flatten()).argsort()[:k]
    ranklist = items[item_locations] # get top recommended items according to predictions
    hr = getHitRatio(ranklist, gtItem)  # if the item recommend was part of the test set, return 1 - success
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def get_hit_ratio_shilling(ranklist, shilling_item_ids):
    for item in shilling_item_ids:
        if item in ranklist:
            return 1
    return 0


def get_ndcg_shilling(ranklist, shilling_item_ids):
    for i in range(len(ranklist)):
        if ranklist[i] in shilling_item_ids:
            return math.log(2) / math.log(i + 2)
    return 0


def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        if ranklist[i] == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
