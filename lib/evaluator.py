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

    def naive_split(self, test_percentage=0.2):
        """
        Split the ratings into test and train data.

        :param float test_percentage: The ratio of the testing data from all the data.
        :returns: a tuple of train and test data.
        :rtype: tuple
        """
        test = numpy.zeros(self.ratings.shape)
        train = self.ratings.copy()
        # TODO split in a more intelligent way
        for user in range(self.ratings.shape[0]):
            non_zeros = self.ratings[user, :].nonzero()[0]
            test_ratings = numpy.random.choice(non_zeros,
                                               size=int(test_percentage * len(non_zeros)))
            train[user, test_ratings] = 0.
            test[user, test_ratings] = self.ratings[user, test_ratings]
        assert(numpy.all((train * test) == 0))
        self.ratings = train
        return train, test

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

    def evaluate(self, n_recommendations, predictions):
        """
        The method calculates the average recall of all users by only looking at the top n_recommendations
        and the normalized Discounted Cumulative Gain.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: Recall at n_recommendations, nDCG for n_recommendations
        :rtype: tuple (float recall, float nDCG)
        """
        recalls = []
        ndcgs = []
        for user in range(self.ratings.shape[0]):
            # loop through all users
            top_recommendations = TopRecommendations(n_recommendations)
            ctr = 0
            liked_items = 0
            # loop through all user predictions
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
            # nDCG
            dcg = 0
            idcg = 0
            i = 0
            j = 0
            for index in list(reversed(top_recommendations.get_indices())):
                dcg += predictions[user, index] / math.log(i + 2, 2)
                i += 1
            for rating in sorted(self.ratings[user, :], reverse=True):
                idcg += rating / math.log(j + 2, 2)
                j += 1
                if j == n_recommendations :
                    break
            ndcgs.append(dcg / idcg)
        return numpy.mean(recalls), numpy.mean(ndcgs)
