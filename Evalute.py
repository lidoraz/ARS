"""
Based on Apr 15, 2016 Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
"""
import math
import heapq  # for retrieval topK
import numpy as np

"""
Evaluation class for MovieLens datasets
Metrics are Hit Ratio and NDCG, based on 10 Most recommended items.
Please note that batch size is 100 for the model.
"""
class Evaluate():
    def __init__(self, model, testRatings, testNegatives, K):
        self.model = model
        self.testRatings = testRatings
        self.testNegatives = testNegatives
        self.K = K

    def evaluate_model(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        # Single thread
        for idx in range(len(self.testRatings)):
            (hr, ndcg) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (hits, ndcgs)


    def eval_one_rating(self, idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        u = rating[0]  # user
        gtItem = rating[1]  # item rated
        items.append(gtItem)  # add the rated item to the list of items.
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u, dtype='int32')  # creates an array of size len(items) with values of u)
        predictions = self.model.predict([users, np.array(items)],
                                     batch_size=100,
                                     verbose=0)  # given user i and set of items, return probability for each of #batch_size items
        for i in range(len(items)):  # construct an item->score map
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)  # get 10 most probable items
        hr = self.getHitRatio(ranklist, gtItem)  # if the item recommend was part of the test set, return 1 - success
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        if gtItem in ranklist:
            return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            if ranklist[i] == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0
