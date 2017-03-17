#!/usr/bin/env python
import numpy
import unittest
from lib.abstract_recommender import AbstractRecommender
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from util.data_parser import DataParser
from util.model_initializer import ModelInitializer


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 30, 4
        documents_cnt, users_cnt = self.documents, self.users
        self.n_factors = 5
        self.n_iterations = 20
        self.k_folds = 3
        self.hyperparameters = {'n_factors': self.n_factors, '_lambda': 0.01}
        self.options = {'k_folds': self.k_folds, 'n_iterations': self.n_iterations}
        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iterations)

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        self.evaluator = Evaluator(self.ratings_matrix)
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)


class TestALS(TestcaseBase):
    def runTest(self):
        cf = CollaborativeFiltering(self.initializer, self.evaluator, self.hyperparameters,
                                    self.options, load_matrices=False)
        print(cf.get_ratings())
        print(cf.get_predictions())
