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
        self.n_recommendations = 8

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
        self.rounded_predictions = (collaborative_filtering.rounded_predictions())


class TestEvaluator(TestcaseBase):
    def runTest(self):
        numpy.set_printoptions(precision=4)
        evaluator = Evaluator(self.ratings_matrix)

        # self.ratings_matrix[3,6] = 0
        # self.ratings_matrix[0,6] = 0

        self.assertEqual(self.predictions.shape, self.ratings_matrix.shape)
        recall_at_x = evaluator.recall_at_x(self.n_recommendations, self.predictions)
        print("recall", recall_at_x)

        # if predictions are  perfect
        if recall_at_x == 1:
            for row in range(self.users):
                for col in range(self.documents):
                    self.assertEqual(self.rounded_predictions[row, col], self.ratings_matrix[row, col])

        ndcg = evaluator.calculate_ndcg(self.n_recommendations, self.predictions)
        print("ndcg", ndcg)
        mrr = evaluator.calculate_mrr(self.n_recommendations, self.predictions)
        print("mrr", mrr)

