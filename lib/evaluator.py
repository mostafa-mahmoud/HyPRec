#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
from sklearn.metrics import mean_squared_error
from util.top_recommendations import TopRecommendations


class Evaluator(object):

    def __init__(self, ratings, abstracts=None):
        """
        Initialize an evaluator array with the initial actual ratings
        matrix
        @param (int[][]) a numpy array containing the initial ratings
        @param (list[str]) a list of the abstracts.
        """
        self.ratings = ratings
        if abstracts:
            self.abstracts = abstracts

    def get_rmse(self, predicted, actual=None):
        """
        The method given a prediction matrix returns the root mean
        squared error (rmse)
        @param (float[][]) numpy matrix of floats representing
        the predicted ratings
        @returns (float) root mean square error
        """
        if actual is None:
            actual = self.ratings
        return numpy.sqrt(mean_squared_error(predicted, actual))

    def calculate_recall(self, ratings, predictions):
        """
        The method given original ratings and predictions returns the recall of the recommender
        @param (int[][]) ratings matrix
        @param (int[][]) predictions matrix (only 0s or 1s)
        @returns (float) recall, ranges from 0 to 1
        """
        denom = sum(sum(ratings))
        nonzeros = ratings.nonzero()
        nonzeros_predictions = predictions[nonzeros]
        return sum(nonzeros_predictions) / denom

    def recall_at_x(self, n_recommendations, predictions):
        """
        The method calculates the average recall of all users by only looking at the top n_recommendations
        @param (int) n_recommendations number of recommendations to look at, sorted by relevance.
        @param (float[][]) predictions calculated predictions of the recommender
        """
        recalls = []
        for user in range(self.ratings.shape[0]):
            top_recommendations = TopRecommendations(n_recommendations)
            ctr = 0
            liked_items = 0
            for rating in predictions[user, :]:
                top_recommendations.insert(ctr, rating)
                liked_items += rating
                ctr += 1
            recommendation_hits = 0
            user_likes = self.ratings[user].sum()
            for index in top_recommendations.get_indices():
                recommendation_hits += self.ratings[user][index]
            recall = recommendation_hits / (min(n_recommendations, user_likes) * 1.0)
            recalls.append(recall)
        return numpy.mean(recalls)
