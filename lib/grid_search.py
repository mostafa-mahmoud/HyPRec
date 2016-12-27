#!/usr/bin/env python
"""
A module that provides functionalities for grid search
will be used for hyperparameter optimization
"""
import collaborative_filtering as cf
import numpy
import itertools as it
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from lib import collaborative_filtering as cf
from util import data_parser as dp
from lib import evaluator as ev


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
        self.recommender = recommender
        self.hyperparameters = hyperparameters
        self.evaluator = ev.Evaluator(recommender.get_ratings())

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
        best_params = dict()
        train, test = self.recommender.split()
        for config in self.get_all_permutations():
            print("running config ")
            print(config)
            self.recommender.set_config(config)
            self.recommender.train()
            error = self.evaluator.get_rmse(self.recommender.get_predictions(), test)
            print('Error: %f' % error)
            if error < best_error:
                best_params = config
        return best_params


if __name__ == "__main__":

    hyperparameters = {
        '_lambda': [0, 0.01, 0.1, 0.5, 10, 100],
        'n_factors': [20, 40, 100, 200, 300]
    }
    R = numpy.array(dp.DataParser.get_ratings_matrix())
    evaluator = ev.Evaluator(R)
    ALS = cf.CollaborativeFiltering(R, evaluator, {'n_factors': 200, '_lambda': 0.1}, True)
    GS = GridSearch(ALS, hyperparameters)
    best_params = GS.train()
    print("best params")
    print(best_params)