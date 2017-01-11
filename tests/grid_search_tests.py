#!/usr/bin/env python
import numpy
import unittest
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from lib.grid_search import GridSearch
from util.data_parser import DataParser
from util.model_initializer import ModelInitializer


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 8, 10
        documents_cnt, users_cnt = self.documents, self.users
        self.hyperparameters = {
            '_lambda': [0.0001, 0.1],
            'n_factors': [10, 20]
        }
        self.n_iterations = 15
        self.initial_config = {
            '_lambda': 0,
            'n_factors': 10
        }
        self.initializer = ModelInitializer(self.initial_config.copy(), self.n_iterations)

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)


class TestGridSearch(TestcaseBase):
    def runTest(self):
        evaluator = Evaluator(self.ratings_matrix)
        cf = CollaborativeFiltering(self.initializer, self.n_iterations,
                                    self.ratings_matrix, evaluator, self.initial_config, load_matrices=True)
        grid_search = GridSearch(cf, self.hyperparameters, False)
        self.checkKeyGenerator(grid_search)
        self.checkCombinationsGenerator(grid_search)
        self.checkGridSearch(grid_search)

    def checkKeyGenerator(self, grid_search):
        key = grid_search.get_key(self.initial_config)
        other_config = {
            'n_factors': 10,
            '_lambda': 0
        }
        other_key = grid_search.get_key(other_config)
        self.assertEqual(key, other_key)
        self.assertEqual(key, '_lambda:0,n_factors:10')

    def checkCombinationsGenerator(self, grid_search):
        combinations = grid_search.get_all_combinations()
        self.assertTrue(isinstance(combinations, list))
        random_config = combinations[int(len(combinations) * numpy.random.random())]
        self.assertTrue(isinstance(random_config, dict))
        self.assertEqual(random_config.keys(), self.hyperparameters.keys())

    def checkGridSearch(self, grid_search):
        best_params = grid_search.train()
        self.assertTrue(isinstance(best_params, dict))
