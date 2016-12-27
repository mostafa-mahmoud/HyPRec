#!/usr/bin/env python
"""
A module that provides functionalities for grid search
will be used for hyperparameter optimization
"""
import collaborative_filtering as cf
import numpy
import itertools as it

class GridSearch(object):

    def __init__(self,
                recommender,
                hyperparameters):
        """
        Train number of recommenders using UV decomposition
        using different parameters.

        Params
        =====
        _lambda: (list) containing different regularization
        parameters

        n_factors: (list) containing different number of latent
        factors to be used in factorizing the ratings matrix
        """
        self._lambda = _lambda
        self.n_factors = n_factors
        self.recommender = recommender
        self.evaluator = Evaluator(recommender.get_ratings())

    def get_all_permutations(self):
        """
        the method retuns all possible permutations
        of the hyperparameters
        @returns (dict[]) array of dicts containing all
        permutations
        """
        names = sorted(self.hyperparameters)
        return [dict(zip(names, prod)) for prod in it.product(
            *(self.hyperparameters[name] for name in names))]

    def train(self):
        """
        The method loops on all hyperparameters and calls
        the train and split method on the recommender then
        calls the get_rmse from the evaluator. it returns
        the best hyperparameters.
        """
        best_error = numpy.inf
        keys = list(self.hyperparameters.keys())
        train, test = recommender.split()
        best_params = dict()
        for config in self.get_all_permutations():
            recommender.set_config(config)
            recommender.train()
            error = evaluator.get_rmse(recommedner.get_predictions)
            if error < best_error:
                best_params = config
        return best_params


