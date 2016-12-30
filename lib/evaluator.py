#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
from sklearn.metrics import mean_squared_error


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
