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
        self.assertEqual(cf.n_factors, self.n_factors)
        self.assertEqual(cf.n_items, self.documents)
        cf.train()
        self.assertEqual(cf.get_predictions().shape, (self.users, self.documents))
        self.assertTrue(isinstance(cf, AbstractRecommender))
        shape = (self.users, self.documents)
        ratings = cf.get_ratings()
        self.assertLessEqual(numpy.amax(ratings), 1 + 1e-6)
        self.assertGreaterEqual(numpy.amin(ratings), -1e-6)
        self.assertTrue(ratings.shape == shape)
        rounded_predictions = cf.rounded_predictions()
        self.assertLessEqual(numpy.amax(rounded_predictions), 1 + 1e-6)
        self.assertGreaterEqual(numpy.amin(rounded_predictions), -1e-6)
        self.assertTrue(rounded_predictions.shape == shape)
        recall = cf.evaluator.calculate_recall(ratings, cf.get_predictions())
        self.assertTrue(-1e-6 <= recall <= 1 + 1e-6)
        random_user = int(numpy.random.random() * self.users)
        random_item = int(numpy.random.random() * self.documents)
        random_prediction = cf.predict(random_user, random_item)
        self.assertTrue(isinstance(random_prediction, numpy.float64))

        train, test = cf.naive_split()
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))

        train_indices, test_indices = cf.get_kfold_indices()
        # k = 3
        first_fold_indices = train_indices[0::self.k_folds], test_indices[0::self.k_folds]
        second_fold_indices = train_indices[1::self.k_folds], test_indices[1::self.k_folds]
        third_fold_indices = train_indices[2::self.k_folds], test_indices[2::self.k_folds]

        train1, test1 = cf.generate_kfold_matrix(first_fold_indices[0], first_fold_indices[1])
        train2, test2 = cf.generate_kfold_matrix(second_fold_indices[0], second_fold_indices[1])
        train3, test3 = cf.generate_kfold_matrix(third_fold_indices[0], third_fold_indices[1])

        total_ratings = numpy.count_nonzero(self.ratings_matrix)

        # ensure that each fold has 1/k of the total ratings
        k_inverse = round(1 / self.k_folds, 1)

        self.assertEqual(k_inverse, numpy.count_nonzero(test1) / total_ratings)

        self.assertEqual(k_inverse, numpy.count_nonzero(test2) / total_ratings)

        self.assertEqual(k_inverse, numpy.count_nonzero(test2) / total_ratings)

        # assert that the folds don't intertwine
        self.assertTrue(numpy.all((train1 * test1) == 0))
        self.assertTrue(numpy.all((train2 * test2) == 0))
        self.assertTrue(numpy.all((train3 * test3) == 0))
        # assert that test sets dont contain the same elements
        self.assertTrue(numpy.all((test1 * test2) == 0))
        self.assertTrue(numpy.all((test2 * test3) == 0))
        self.assertTrue(numpy.all((test1 * test3) == 0))
