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
        self.n_recommendations = 3

        def mock_get_ratings_matrix(self=None):

            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)

        evaluator = Evaluator(self.ratings_matrix)
        config = {'n_factors': 5, '_lambda': 0.01}
        collaborative_filtering = CollaborativeFiltering(self.ratings_matrix, evaluator, config)
        collaborative_filtering.train()
        self.predictions = (collaborative_filtering.get_predictions())


class TestEvaluator(TestcaseBase):
    def runTest(self):

        evaluator = Evaluator(self.ratings_matrix)
        print(self.ratings_matrix)
        self.assertEqual(self.predictions.shape, self.ratings_matrix.shape)
        recall_at_k, nDCG = evaluator.evaluate(self.n_recommendations, self.predictions)

        # if predictions are  perfect
        if recall_at_k == 1:
            for row in range(self.users):
                for col in range(self.documents):
                    self.assertEqual(self.predictions[row, col], self.ratings_matrix[row, col])

        train, test = evaluator.naive_split(test_percentage=0.2)
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))

        evaluator.ratings = numpy.ones(self.ratings_matrix.shape)
        train, test = evaluator.naive_split(test_percentage=0.2)
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         self.ratings_matrix.shape[0] * self.ratings_matrix.shape[1])
