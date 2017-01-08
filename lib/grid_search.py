#!/usr/bin/env python
"""
A module that provides functionalities for grid search
will be used for hyperparameter optimization.
"""

import numpy
import itertools as it
from lib.evaluator import Evaluator


class GridSearch(object):

    def __init__(self, recommender, hyperparameters, verbose=True):
        """
        Train number of recommenders using UV decomposition using different parameters.

        :param AbstractRecommender recommender:
        :param dict hyperparameters: A dictionary of the hyperparameters.
        """
        self.recommender = recommender
        self.hyperparameters = hyperparameters
        self._v = verbose
        self.evaluator = Evaluator(recommender.get_ratings())
        self.all_errors = dict()

    def get_all_combinations(self):
        """
        The method retuns all possible combinations of the hyperparameters.

        :returns: array of dicts containing all combinations
        :rtype: list[dict]

        >>> get_all_combinations({'_lambda': [0, 0.1], 'n_factors': [20, 40]})
        [{'n_factors': 20, '_lambda': 0}, {'n_factors': 40, '_lambda': 0},
        {'n_factors': 20, '_lambda': 0.1}, {'n_factors': 40, '_lambda': 0.1}]
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
        best_params = dict()
        train, test = self.recommender.split()
        for config in self.get_all_combinations():
            if self._v:
                print("running config ")
                print(config)
            self.recommender.set_config(config)
            self.recommender.train()
            rounded_predictions = self.recommender.rounded_predictions()
            test_recall = self.evaluator.calculate_recall(test, rounded_predictions)
            train_recall = self.evaluator.calculate_recall(self.recommender.get_ratings(), rounded_predictions)
            if self._v:
                print('Train error: %f, Test error: %f' % (train_recall, test_recall))
            if 1 - test_recall < best_error:
                best_params = config
                best_error = 1 - test_recall
            current_key = self.get_key(config)
            self.all_errors[current_key] = dict()
            self.all_errors[current_key]['train_recall'] = train_recall
            self.all_errors[current_key]['test_recall'] = test_recall
        return best_params

    def get_key(self, config):
        """
        Given a dict (config) the function generates a key that uniquely represents
        this config to be used to store all errors

        :param dict config: given configuration.
        :returns: string reperesenting the unique key of the configuration
        :rtype: str

        >>> get_key({n_iter: 1, n_factors:200})
        'n_iter:1,n_factors:200'
        """
        generated_key = ''
        keys_array = sorted(config)
        for key in keys_array:
            generated_key += key + ':'
            generated_key += str(config[key]) + ','
        return generated_key.strip(',')

    def get_all_errors(self):
        """
        The method returns all errors calculated for every configuration.

        :returns: containing every single computed test error.
        :rtype: dict
        """
        return self.all_errors
