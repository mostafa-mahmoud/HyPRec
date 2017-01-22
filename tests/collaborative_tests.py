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
        self.documents, self.users = 8, 10
        documents_cnt, users_cnt = self.documents, self.users
        self.config = {'n_factors': 5, '_lambda': 0.01}
        self.n_iterations = 15
        self.initializer = ModelInitializer(self.config.copy(), self.n_iterations)

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        self.k = 3
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)


class TestALS(TestcaseBase):
    def runTest(self):
        evaluator = Evaluator(self.ratings_matrix)
        cf = CollaborativeFiltering(self.initializer, self.n_iterations,
                                    self.ratings_matrix, evaluator, self.config, load_matrices=True)
        self.assertEqual(cf.n_factors, 5)
        self.assertEqual(cf.n_items, self.documents)
        cf.train()
        self.assertEqual(cf.get_predictions().shape, (self.users, self.documents))
        self.assertTrue(isinstance(cf, AbstractRecommender))
        shape = (self.users, self.documents)
        ratings = cf.get_ratings()
        self.assertTrue(numpy.amax(ratings <= 1))
        self.assertTrue(numpy.amin(ratings >= 0))
        self.assertTrue(ratings.shape == shape)
        rounded_predictions = cf.rounded_predictions()
        self.assertTrue(numpy.amax(rounded_predictions <= 1))
        self.assertTrue(numpy.amin(rounded_predictions >= 0))
        self.assertTrue(rounded_predictions.shape == shape)
        recall = evaluator.calculate_recall(ratings, cf.get_predictions())
        self.assertTrue(0 <= recall <= 1)
        random_user = int(numpy.random.random() * self.users)
        random_item = int(numpy.random.random() * self.documents)
        random_prediction = cf.predict(random_user, random_item)
        self.assertTrue(isinstance(random_prediction, numpy.float64))

        train, test = cf.naive_split()
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))
        train, test = cf.naive_split(docs=True)
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))

        train, test = cf.get_kfold_indices(3)
        for train_index_list, test_index_list in zip(test, train):
            self.assertFalse(numpy.in1d(train_index_list.all(), test_index_list.all()))

        # Training one more iteration always reduces the rmse.
        additional_iterations = 5
        initial_rmse = evaluator.get_rmse(cf.get_predictions())
        cf.set_iterations(1)
        for i in range(additional_iterations):
            cf.partial_train()
            final_rmse = evaluator.get_rmse(cf.get_predictions())
            self.assertTrue(initial_rmse >= final_rmse)
            initial_rmse = final_rmse
