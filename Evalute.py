"""
Based on Apr 15, 2016 Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16
Evaluate the performance of Top-k recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
"""
import math
import heapq  # for retrieval topk
import numpy as np

"""
Evaluation for MovieLens datasets
Metrics are Hit Ratio and NDCG, based on 10 Most recommended items.
*NOTE: These two metrics focus on RECOMMENDATION tools
Please note that batch size is 100 for the model.

    Evaluate the performance (Hit_Ratio, NDCG) of top-k recommendation
    Return: score of each test rating.
"""

from time import time

from keras.models import clone_model

# TODO: MODEL_TAKE_BEST? HOW TO MAKE IT?
def baseline_train_evalute_model(model, train_set, test_set, batch_size=512, epochs=5):
    best_hr = 0
    best_ndcg = 0
    best_epoch = 0
    models = []
    for epoch in range(epochs):
        t1 = time()
        (user_input, item_input, labels) = train_set
        loss = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
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


def pert_train_evaluate_model(model, train_set, test_set, batch_size=512, epochs=5, pert_model_take_best= False, verbose= 0):
    best_hr = 0
    best_ndcg = 0
    best_epoch = 0
    for epoch in range(epochs):
        t1 = time()
        (user_input, item_input, labels) = train_set
        loss = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        mean_hr, mean_ndcg, time_eval = evaluate_model(model, test_set, verbose=0)
        if verbose > 1:
            print('Iteration: %d Fit:[%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, Eval:[%.1f s]'
                  % (epoch + 1, t2 - t1, mean_hr, mean_ndcg, loss.history['loss'][0], time_eval))
        # TODO: CHANGED HERE FOR ATLEAST 2 EPOCHS, takeing the first epoch does not change because learning rate is small

        if pert_model_take_best:
            if mean_hr > best_hr and epoch > 0:
            # if mean_hr > best_hr or mean_ndcg > best_ndcg and epoch > 0:
                best_hr = mean_hr
                best_ndcg = mean_ndcg
                best_epoch = epoch
        else:
            best_hr = mean_hr
            best_ndcg = mean_ndcg
            best_epoch = epoch

    return best_epoch, best_hr, best_ndcg

import matplotlib.pyplot as plt
def plot(HR_list, NDCG_list):
    epochs = list(range(len(HR_list)))
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, HR_list)
    plt.xlabel('Epoch')
    plt.ylabel('Hit Rate')
    plt.title(f'Adv Attack: Metrics drop over {(len(HR_list))} epochs')
    ax2 = plt.twinx()
    ax2.set_ylabel('NDCG', color='tab:orange')
    ax2.plot(epochs, NDCG_list, color='orange')
    plt.show()


def evaluate_model(model, test_set, k= 10, verbose=1):
    t0 = time()
    user_list = list(test_set.keys())
    negatives_list = list(test_set.values())
    hits, ndcgs = [], []
    times = []
    for idx in range(len(user_list)):     # Single thread
        t0 = time()
        (hr, ndcg) = eval_one_rating(model, idx, user_list, negatives_list, k)
        t1 = time()
        times.append(t1-t0)
        hits.append(hr)
        ndcgs.append(ndcg)
    mean_hr, mean_ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    time_eval = time() - t0
    if verbose > 1:
        print('Init: HR = %.4f, NDCG = %.4f, Eval:[%.2f s]'
              % (mean_hr, mean_ndcg, time_eval))
    return mean_hr, mean_ndcg, time_eval


def eval_one_rating(model, idx, user_list, negatives_list, k):
    user = user_list[idx]

    gtItem = negatives_list[idx][-1]

    map_item_score = {}
    user_arr = np.full(len(negatives_list[idx]), user, dtype='int32')  # creates an array of size len(items) with values of u)
    items = np.array(negatives_list[idx])
    predictions = model.predict([user_arr, items],
                                 batch_size=100,
                                 verbose=0)  # given user i and set of items, return probability for each of #batch_size items
    for i in range(len(items)):  # construct an item->score map
        item = items[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    # equivalent: sorted(map_item_score.items(), key=itemgetter(1), reverse=True)[:k]
    ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)  # get 10 most probable items
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
