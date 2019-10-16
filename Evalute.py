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
Please note that batch size is 100 for the model.

    Evaluate the performance (Hit_Ratio, NDCG) of top-k recommendation
    Return: score of each test rating.
"""


def evaluate_model(model, test_set, k= 10):

    user_list = list(test_set.keys())
    negatives_list = list(test_set.values())
    hits, ndcgs = [], []
    # Single thread
    for idx in range(len(user_list)):
        (hr, ndcg) = eval_one_rating(model, idx, user_list, negatives_list, k)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()


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

def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        if ranklist[i] == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
