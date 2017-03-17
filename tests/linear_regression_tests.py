#!/usr/bin/env python
import numpy
import unittest
from lib.abstract_recommender import AbstractRecommender
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from lib.linear_regression import LinearRegression
from util.data_parser import DataParser
from util.model_initializer import ModelInitializer


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 10, 50
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
                                    self.options, load_matrices=False, is_hybrid=False)

        cf.train()
        ratings = cf.get_ratings()

        train_data = cf.train_data
        test_data = cf.test_data
        cf_predictions = cf.get_predictions()


        recall_without_lr = self.evaluator.calculate_recall(self.ratings_matrix, cf_predictions)

        mock_bad_predictions = (numpy.random.randint(2, size=(self.users, self.documents)))

        linearRegressor = LinearRegression(train_data, test_data, mock_bad_predictions, cf_predictions)

        # ensure all matrices are flattened
        self.assertEquals(linearRegressor.flat_item_based_ratings.shape[0], self.users * self.documents)
        self.assertEquals(linearRegressor.flat_collaborative_ratings.shape[0], self.users * self.documents)
        self.assertEquals(linearRegressor.flat_train_labels.shape[0], self.users * self.documents)
        self.assertEquals(linearRegressor.flat_test_labels.shape[0], self.users * self.documents)

        # Because our predictions from the second recommender are random there will be a lot of bad predictions
        # recall should lower significantly after LR.

        recall_with_lr = self.evaluator.calculate_recall(self.ratings_matrix, linearRegressor.train())
        print(recall_with_lr)
        self.assertTrue(recall_with_lr < recall_without_lr)
