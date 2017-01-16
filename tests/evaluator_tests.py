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

        self.n_recommendations = 1

        def mock_get_ratings_matrix(self=None):

            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)

        evaluator = Evaluator(self.ratings_matrix)
        config = {'n_factors': 5, '_lambda': 0.01}
        collaborative_filtering = CollaborativeFiltering(self.initializer, self.n_iterations,
                                    self.ratings_matrix, evaluator, self.config, load_matrices=True)
        collaborative_filtering.train()
        self.predictions = (collaborative_filtering.get_predictions())
        self.rounded_predictions = (collaborative_filtering.rounded_predictions())


class TestEvaluator(TestcaseBase):
    def runTest(self):
        unmodified_rating_matrix = self.ratings_matrix.copy()

        numpy.set_printoptions(precision=4)
        evaluator = Evaluator(self.ratings_matrix)

        self.assertEqual(self.predictions.shape, self.ratings_matrix.shape)
        recall_at_x = evaluator.recall_at_x(self.n_recommendations, self.predictions)

        # if predictions are  perfect
        if recall_at_x == 1:
            for row in range(self.users):
                for col in range(self.documents):
                    self.assertEqual(self.rounded_predictions[row, col], self.ratings_matrix[row, col])

        # If we modify all the top predictions for half the users,
        # recall should be 0.5 by definition
        for i in range(0, self.users, 2):
           self.ratings_matrix[i,(numpy.argmax(self.predictions[i], axis=0))] = 0
        recall_at_x = evaluator.recall_at_x(self.n_recommendations, self.predictions)
        self.assertEqual(0.5, recall_at_x)
        
        # restore the unmodified rating matrix
        self.ratings_matrix = unmodified_rating_matrix.copy()
        evaluator = Evaluator(self.ratings_matrix)
        
        for i in range(0, self.users):
           self.ratings_matrix[i,(numpy.argmax(self.predictions[i], axis=0))] = 0
        ndcg = evaluator.calculate_ndcg(self.n_recommendations, self.predictions)
        self.assertEqual(0.0, ndcg)


        self.ratings_matrix = unmodified_rating_matrix.copy()
        # mrr will always decrease as we set the highest prediction's index
        # to 0 in the rating matrix. top_n recommendations set to 0.
        evaluator = Evaluator(self.ratings_matrix)
        mrr = []
        for i in range(self.users):
            mrr.append(evaluator.calculate_mrr(self.n_recommendations, self.predictions))
            self.ratings_matrix[i,(numpy.argmax(self.predictions[i], axis=0))] = 0
            if i > 1:
                self.assertTrue(mrr[i] <  mrr[i-1])





