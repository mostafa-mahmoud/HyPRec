#!/usr/bin/env python
import numpy
import unittest
from lib.abstract_recommender import AbstractRecommender
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from util.data_parser import DataParser


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 8, 10
        documents_cnt, users_cnt = self.documents, self.users

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)


class TestALS(TestcaseBase):
    def runTest(self):
        evaluator = Evaluator(self.ratings_matrix)
        config = {'n_factors': 5, '_lambda': 0.01}
        collaborative_filtering = CollaborativeFiltering(self.ratings_matrix, evaluator, config)
        self.assertEqual(collaborative_filtering.n_factors, 5)
        self.assertEqual(collaborative_filtering.n_items, self.documents)
        collaborative_filtering.train()
        self.assertEqual(collaborative_filtering.get_predictions().shape, (self.users, self.documents))
        self.assertTrue(isinstance(collaborative_filtering, AbstractRecommender))
        shape = (self.users, self.documents)
        ratings = collaborative_filtering.get_ratings()
        self.assertTrue(numpy.amax(ratings <= 1))
        self.assertTrue(numpy.amin(ratings >= 0))
        self.assertTrue(ratings.shape == shape)
        rounded_predictions = collaborative_filtering.rounded_predictions()
        self.assertTrue(numpy.amax(rounded_predictions <= 1))
        self.assertTrue(numpy.amin(rounded_predictions >= 0))
        self.assertTrue(rounded_predictions.shape == shape)
        recall = evaluator.calculate_recall(ratings, collaborative_filtering.get_predictions())
        self.assertTrue(0 <= recall <= 1)
        random_user = int(numpy.random.random() * self.users)
        random_item = int(numpy.random.random() * self.documents)
        random_prediction = collaborative_filtering.predict(random_user, random_item)
        self.assertTrue(isinstance(random_prediction, numpy.float64))
        train, test = collaborative_filtering.naive_split(test_percentage=0.2)
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))
        self.assertTrue(numpy.all((train * test) == 0))



        # Training one more iteration always reduces the rmse.
        additional_iterations = 5
        initial_rmse = evaluator.get_rmse(collaborative_filtering.get_predictions())
        for i in range(additional_iterations):
            collaborative_filtering.partial_train(1)
            final_rmse = evaluator.get_rmse(collaborative_filtering.get_predictions())
            self.assertTrue(initial_rmse >= final_rmse)
            initial_rmse = final_rmse
