"""
A module that provides functionalities for grid search
will be used for hyperparameter optimization
"""

import sys
import os
import numpy
import itertools as it
from lib.evaluator import Evaluator


class GridSearch(object):

    def __init__(self, recommender, hyperparameters):
        """
        Train number of recommenders using UV decomposition
        using different parameters.
        @param (object) recommender
        @param (dict) hyperparameters, list of the hyperparameters
        """
        self.recommender = recommender
        self.hyperparameters = hyperparameters
        self.evaluator = Evaluator(recommender.get_ratings())

    def get_all_combinations(self):
        """
        the method retuns all possible combinations of the hyperparameters
        Example: hyperparameters = {'_lambda': [0, 0.1], 'n_factors': [20, 40]}
        Output: [{'n_factors': 20, '_lambda': 0}, {'n_factors': 40, '_lambda': 0},
        {'n_factors': 20, '_lambda': 0.1}, {'n_factors': 40, '_lambda': 0.1}]
        @returns (dict[]) array of dicts containing all combinations
        """
        names = sorted(self.hyperparameters)
        return [dict(zip(names, prod)) for prod in it.product(
            *(self.hyperparameters[name] for name in names))]

    def train(self):
        """
        The method loops on all  possible combinations of hyperparameters and calls
        the train and split method on the recommender. the train and test errors are
        saved and the hyperparameters that produced the best test error are returned
        """
        best_error = numpy.inf
        keys = list(self.hyperparameters.keys())
        best_params = dict()
        train, test = self.recommender.split()
        for config in self.get_all_combinations():
            print("running config ")
            print(config)
            self.recommender.set_config(config)
            self.recommender.train()
            error = self.evaluator.get_rmse(self.recommender.get_predictions(), test)
            print('Error: %f' % error)
            if error < best_error:
                best_params = config
        return best_params


