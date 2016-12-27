#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
from sklearn.metrics import mean_squared_error


class Evaluator(object):

    def __init__(self, ratings):
        """
        Initialize an evaluator array with the initial actual ratings
        matrix
        @param (int[][]) a numpy array containing the initial ratings
        """
        self.ratings = ratings

    def get_rmse(predicted):
        """
        The method given a prediction matrix returns the root mean
        squared error (rmse)
        @param (float[][]) numpy matrix of floats representing
        the predicted ratings
        @returns (float) root mean square error
        """
        return numpy.sqrt(mean_squared_error(predicted, self.ratings))
