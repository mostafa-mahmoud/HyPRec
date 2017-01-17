#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
import math
from sklearn.metrics import mean_squared_error
from util.top_recommendations import TopRecommendations


class Evaluator(object):

    def __init__(self, ratings, abstracts_preprocessor=None):
        """
        Initialize an evaluator array with the initial actual ratings matrix.

        :param int[][] ratings: A numpy array containing the initial ratings.
        :param AbstractsPreprocessor abstracts_preprocessor: A list of the abstracts.
        """
        self.ratings = ratings
        if abstracts_preprocessor:
            self.abstracts_preprocessor = abstracts_preprocessor

    def get_rmse(self, predicted, actual=None):
        """
        The method given a prediction matrix returns the root mean squared error (rmse).

        :param float[][] predicted: numpy matrix of floats representing the predicted ratings
        :returns: root mean square error
        :rtype: float
        """
        if actual is None:
            actual = self.ratings
        return numpy.sqrt(mean_squared_error(predicted, actual))

    def calculate_recall(self, ratings, predictions):
        """
        The method given original ratings and predictions returns the recall of the recommender

        :param int[][] ratings: ratings matrix
        :param int[][] predictions: predictions matrix (only 0s or 1s)
        :returns: recall, ranges from 0 to 1
        :rtype: float
        """
        denom = sum(sum(ratings))
        nonzeros = ratings.nonzero()
        nonzeros_predictions = predictions[nonzeros]
        return sum(nonzeros_predictions) / denom

    def recall_at_x(self, n_recommendations, predictions):
        """
        The method calculates the average recall of all users by only looking at the top n_recommendations
        and the normalized Discounted Cumulative Gain.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: Recall at n_recommendations
        :rtype: numpy.float16
        """
        recalls = []
        for user in range(self.ratings.shape[0]):
            top_recommendations = TopRecommendations(n_recommendations)
            for ctr, rating in enumerate(predictions[user, :]):
                top_recommendations.insert(ctr, rating)
            recommendation_hits = 0
            user_likes = self.ratings[user].sum()
            for index in list(reversed(top_recommendations.get_indices())):
                recommendation_hits += self.ratings[user][index]
            recall = recommendation_hits / (min(n_recommendations, user_likes) * 1.0)
            recalls.append(recall)
        return numpy.mean(recalls, dtype=numpy.float16)

    def calculate_ndcg(self, n_recommendations, predictions):
        """
        The method calculates the normalized Discounted Cumulative Gain of all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: nDCG for n_recommendations
        :rtype: numpy.float16
        """
        ndcgs = []
        for user in range(self.ratings.shape[0]):
            top_recommendations = TopRecommendations(n_recommendations)
            ctr = 0
            for rating in predictions[user, :]:
                top_recommendations.insert(ctr, rating)
                ctr += 1
            dcg = 0
            idcg = 0
            for pos_index, index in enumerate(list(reversed(top_recommendations.get_indices()))):
                dcg += numpy.power(2, self.ratings[user, index]) - 1 / numpy.log2(pos_index + 2)
            for pos_index, rating in enumerate(sorted(self.ratings[user, :], reverse=True)):
                idcg += numpy.power(2, rating) - 1 / numpy.log2(pos_index + 2)
                if (pos_index + 1) == n_recommendations:
                    break
            ndcgs.append(dcg / idcg)
        return numpy.mean(ndcgs, dtype=numpy.float16)

    def calculate_mrr(self, n_recommendations, predictions):
        """
        The method calculates the mean reciprocal rank for all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: mrr at n_recommendations
        :rtype: numpy.float16
        """

        mrr_list = []

        for user in range(self.ratings.shape[0]):
            top_recommendations = TopRecommendations(n_recommendations)
            ctr = 0
            for rating in predictions[user, :]:
                top_recommendations.insert(ctr, rating)
                ctr += 1
            for mrr_index, index in enumerate(list(reversed(top_recommendations.get_indices()))):
                if self.ratings[user, index] == 1:
                    mrr_list.append(1 / (mrr_index + 1))
                    break
                # if no hit found
                if mrr_index + 1 == n_recommendations:
                    mrr_list.append(0)

        return numpy.mean(mrr_list, dtype=numpy.float16)
