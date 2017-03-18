#!/usr/bin/env python
import numpy
import unittest
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from util.data_parser import DataParser
from util.model_initializer import ModelInitializer


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 18, 10
        documents_cnt, users_cnt = self.documents, self.users
        self.n_iterations = 15
        self.k_folds = 3
        self.hyperparameters = {'n_factors': 5, '_lambda': 0.01}
        self.options = {'n_iterations': self.n_iterations, 'k_folds': self.k_folds}
        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iterations)
        self.n_recommendations = 1

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]

        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)

        self.evaluator = Evaluator(self.ratings_matrix)
        self.cf = CollaborativeFiltering(self.initializer, self.evaluator, self.hyperparameters,
                                         self.options, load_matrices=True)
        self.cf.train()
        self.cf.evaluator.k_folds = self.k_folds
        self.test_data = self.cf.test_data
        self.predictions = self.cf.get_predictions()
        self.rounded_predictions = self.cf.rounded_predictions()


class TestEvaluator(TestcaseBase):
    def runTest(self):
        train, test = self.cf.evaluator.naive_split()
        self.assertEqual(numpy.count_nonzero(train) + numpy.count_nonzero(test),
                         numpy.count_nonzero(self.ratings_matrix))

        train_indices, test_indices = self.cf.evaluator.get_kfold_indices()
        # k = 3
        first_fold_indices = train_indices[0::self.k_folds], test_indices[0::self.k_folds]
        second_fold_indices = train_indices[1::self.k_folds], test_indices[1::self.k_folds]
        third_fold_indices = train_indices[2::self.k_folds], test_indices[2::self.k_folds]

        train1, test1 = self.cf.evaluator.generate_kfold_matrix(first_fold_indices[0], first_fold_indices[1])
        train2, test2 = self.cf.evaluator.generate_kfold_matrix(second_fold_indices[0], second_fold_indices[1])
        train3, test3 = self.cf.evaluator.generate_kfold_matrix(third_fold_indices[0], third_fold_indices[1])

        total_ratings = numpy.count_nonzero(self.ratings_matrix)

        # ensure that each fold has 1/k of the total ratings
        k_inverse =  1 / self.k_folds
        self.assertTrue(abs(k_inverse - ((numpy.count_nonzero(test1)) / total_ratings)) < 1e-6)
        self.assertTrue(abs(k_inverse - ((numpy.count_nonzero(test1)) / total_ratings)) < 1e-6)
        self.assertTrue(abs(k_inverse - ((numpy.count_nonzero(test1)) / total_ratings)) < 1e-6)

        # assert that the folds don't intertwine
        self.assertTrue(numpy.all((train1 * test1) == 0))
        self.assertTrue(numpy.all((train2 * test2) == 0))
        self.assertTrue(numpy.all((train3 * test3) == 0))
        # assert that test sets dont contain the same elements
        self.assertTrue(numpy.all((test1 * test2) == 0))
        self.assertTrue(numpy.all((test2 * test3) == 0))
        self.assertTrue(numpy.all((test1 * test3) == 0))

        evaluator = Evaluator(self.ratings_matrix)
        self.assertEqual(self.predictions.shape, self.ratings_matrix.shape)
        recall = evaluator.calculate_recall(self.ratings_matrix, self.predictions)
        # if predictions are  perfect
        if recall == 1:
            for row in range(self.users):
                for col in range(self.documents):
                    self.assertEqual(self.rounded_predictions[row, col], self.ratings_matrix[row, col])

        # If we modify all the top predictions for half the users,
        # recall should be 0.5 by definition
        for i in range(0, self.users, 2):
            evaluator.ratings[i, self.rounded_predictions[i].nonzero()[0]] = 0
        recall_at_x = evaluator.recall_at_x(self.n_recommendations, self.predictions,
                                            self.ratings_matrix, self.rounded_predictions)
        self.assertEqual(0.5, recall_at_x)
        self.setUp()
        evaluator.ratings = self.ratings_matrix.copy()

        # removing all top hits, should yield ndcg of 0 as number of recs is 1.
        for i in range(0, self.users):
            evaluator.ratings[i] = 0
        ndcg = evaluator.calculate_ndcg(self.n_recommendations, self.predictions,
                                        evaluator.ratings, self.test_data)

        self.assertEqual(0.0, ndcg)

        # restore the unmodified rating matrix
        self.setUp()
        evaluator.ratings = self.ratings_matrix.copy()

        # mrr will always decrease as we set the highest prediction's index
        # to 0 in the rating matrix. top_n recommendations set to 0.
        mrr = []
        for i in range(self.users):
            evaluator.ratings[i, (numpy.argmax(self.predictions[i], axis=0))] = 0
            mrr.append(evaluator.calculate_mrr(self.n_recommendations, self.predictions,
                                               self.rounded_predictions, evaluator.ratings))
            if i > 1:
                self.assertLessEqual(mrr[i], mrr[i-1])
