#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
from sklearn.metrics import mean_squared_error
from util.top_recommendations import TopRecommendations


class Evaluator(object):

    def __init__(self, ratings, abstracts_preprocessor=None):
        """
        Initialize an evaluator array with the initial actual ratings matrix.

        :param int[][] ratings: A numpy array containing the initial ratings.
        :param AbstractsPreprocessor abstracts_preprocessor: A list of the abstracts.
        :param list[][] recommendation_indices: stores recommended indices for each user.
        :param bool recs_loaded: False if recommendations have not been loaded yet and vice versa.
        """
        self.ratings = ratings
        if abstracts_preprocessor:
            self.abstracts_preprocessor = abstracts_preprocessor
        self.recommendation_indices = [[] for i in range(self.ratings.shape[0])]
        self.recs_loaded = False

    def load_top_recommendations(self, n_recommendations, predictions, test_data):
        """
        This method loads the top n recommendations into a local variable.
        :param int n_recommendations: number of recommendations to be generated.
        :param int[][] predictions: predictions matrix (only 0s or 1s)
        """

        for user in range(self.ratings.shape[0]):
            nonzeros = test_data[user].nonzero()[0]
            top_recommendations = TopRecommendations(n_recommendations)
            for index in nonzeros:
                top_recommendations.insert(index, predictions[user][index])
            self.recommendation_indices[user] = list(reversed(top_recommendations.get_indices()))
            top_recommendations = None

        self.recs_loaded = True

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
        return sum(nonzeros_predictions) / denom  # Division by zeros are handled.

    def recall_at_x(self, n_recommendations, predictions, ratings, rounded_predictions):
        """
        The method calculates the average recall of all users by only looking at the top n_recommendations
        and the normalized Discounted Cumulative Gain.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param int[][] ratings: ratings matrix
        :param float[][] predictions: calculated predictions of the recommender.
        :param int[][] test_data: test data.
        :returns: Recall at n_recommendations
        :rtype: numpy.float16
        """

        if self.recs_loaded is False:
            self.load_top_recommendations(n_recommendations, predictions, ratings)

        recalls = []
        for user in range(ratings.shape[0]):
            recommendation_hits = 0
            user_likes = ratings[user].sum()
            recall = 0
            if user_likes != 0:
                for ctr, index in enumerate(self.recommendation_indices[user]):
                    recommendation_hits += ratings[user][index] * rounded_predictions[user][index]
                    if ctr == n_recommendations - 1:
                        break
                recall = recommendation_hits / (min(n_recommendations, user_likes) * 1.0)
            recalls.append(recall)
        return numpy.mean(recalls, dtype=numpy.float16)

    def calculate_ndcg(self, n_recommendations, predictions, test_data, rounded_predictions):
        """
        The method calculates the normalized Discounted Cumulative Gain of all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: nDCG for n_recommendations
        :rtype: numpy.float16
        """

        if self.recs_loaded is False:
            self.load_top_recommendations(n_recommendations, predictions, test_data)
        ndcgs = []
        always_hit = True
        for user in range(self.ratings.shape[0]):
            dcg = 0
            idcg = 0
            for pos_index, index in enumerate(self.recommendation_indices[user]):
                dcg += (self.ratings[user, index] * rounded_predictions[user][index]) / numpy.log2(pos_index + 2)
                idcg += 1 / numpy.log2(pos_index + 2)
                if pos_index + 1 == n_recommendations:
                    break
            if idcg != 0:
                ndcgs.append(dcg / idcg)
        return numpy.mean(ndcgs, dtype=numpy.float16)

    def calculate_mrr(self, n_recommendations, predictions, test_data, rounded_predictions):
        """
        The method calculates the mean reciprocal rank for all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: mrr at n_recommendations
        :rtype: numpy.float16
        """
        if self.recs_loaded is False:
            self.load_top_recommendations(n_recommendations, predictions, test_data)

        mrr_list = [0]

        for user in range(self.ratings.shape[0]):
            for mrr_index, index in enumerate(self.recommendation_indices[user]):
                score = self.ratings[user][index] * rounded_predictions[user][index]
                mrr_list.append(score / (mrr_index + 1))
                if score == 1:
                    break
                if mrr_index + 1 == n_recommendations:
                    break

        return numpy.mean(mrr_list, dtype=numpy.float16)
