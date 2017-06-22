#!/usr/bin/env python
import numpy
import unittest
from numpy import log2
from lib.evaluator import Evaluator


class TestMetrics(unittest.TestCase):
    def mean(self, s):
        return sum(s) / len(s)

    def assertAlmostEqual(self, x, y):
        self.assertTrue(abs(x - y) < 1e-3, '%lf != %lf' % (x, y))

    def setUp(self):
        """
        Setting up the ratings, expected ratings and recommendations.
        The comments are showing where are the matching recommendations.
        A matching recommendation will occur at the recommendation_indcies list,
        and the corresponding ratings and expected rating are both positive.
        """
                                   # 0  1  2  3  4  5  6  7  8
        self.ratings = numpy.array([[1, 1, 0, 0, 1, 0, 1, 0, 0],
                                   #    ^
                                    [0, 0, 1, 1, 0, 0, 0, 1, 0],
                                   #       ^
                                    [1, 1, 0, 1, 0, 0, 1, 0, 1],
                                   #          ^              ^
                                    [1, 0, 0, 0, 1, 0, 0, 0, 0],
                                   # ^
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])
                                   #
                                            # 0  1  2  3  4  5  6  7  8
        self.expected_ratings = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                                            #    ^
                                             [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                            #       ^
                                             [0, 0, 0, 1, 0, 0, 0, 0, 1],
                                            #          ^              ^
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                            # ^
                                             [0, 1, 0, 0, 0, 0, 0, 0, 0]])
                                            #
        self.recommendation_indices = numpy.array([[1],
                                                   # 1 matches -> 1/1
                                                   [3, 2],
                                                   # 3 doesn't match, 2 matches -> 1/2
                                                   [4, 6, 3, 0, 8],
                                                   # 4,6,0 don't match, 3, 8 match -> 1/3, 1/5
                                                   [0],
                                                   # 0 matches -> 1/1
                                                   [0]])
                                                   # no matches -> 0
        self.n_users, self.n_items = self.ratings.shape
        self.evaluator = Evaluator(self.ratings)
        self.evaluator.recs_loaded = True
        self.evaluator.recommendation_indices = self.recommendation_indices


class TestMRR(TestMetrics):
    def runTest(self):
        self.assertAlmostEqual(self.evaluator.calculate_mrr(1, None, self.ratings, self.expected_ratings),
                               (1 / 1 + 1 / 1) / self.n_users)

        self.assertAlmostEqual(self.evaluator.calculate_mrr(4, None, self.ratings, self.expected_ratings),
                               (1 / 1 + 1 / 2 + 1 / 3 + 1 / 1) / self.n_users)

        self.assertAlmostEqual(self.evaluator.calculate_mrr(5, None, self.ratings, self.expected_ratings),
                               (1 / 1 + 1 / 2 + 1 / 3 + 1 / 1) / self.n_users)


class TestNDCG(TestMetrics):
    def get_idcg_k(self, k):
        # TODO: Is it correct to normalize over the minimium of k and recommendations?
        return [sum(1 / log2(i + 1) for i in range(1, 1 + min(k, len(self.recommendation_indices[user]))))
                for user in range(self.n_users)]

    def get_ndcg(self, dcg, idcg):
        return [dcg[user] / idcg[user] for user in range(self.n_users)]

    def runTest(self):
        idcg1 = self.get_idcg_k(1)
        dcg1 = [1 / log2(1 + 1),  # matches 1st
                0,
                0,
                1 / log2(1 + 1),  # matches 1st
                0]
        self.assertAlmostEqual(self.evaluator.calculate_ndcg(1, None, self.ratings, self.expected_ratings),
                               self.mean(self.get_ndcg(dcg1, idcg1)))

        idcg4 = self.get_idcg_k(4)
        dcg4 = [1 / log2(1 + 1),  # matches 1st
                1 / log2(2 + 1),  # matches 2nd
                1 / log2(3 + 1),  # matches 3rd
                1 / log2(1 + 1),  # matches 1st
                0]
        self.assertAlmostEqual(self.evaluator.calculate_ndcg(4, None, self.ratings, self.expected_ratings),
                               self.mean(self.get_ndcg(dcg4, idcg4)))

        idcg5 = self.get_idcg_k(5)
        dcg5 = [1 / log2(1 + 1),                    # matches 1st
                1 / log2(2 + 1),                    # matches 2nd
                1 / log2(3 + 1) + 1 / log2(5 + 1),  # matches 3rd
                1 / log2(1 + 1),                    # matches 1st
                0]
        self.assertAlmostEqual(self.evaluator.calculate_ndcg(5, None, self.ratings, self.expected_ratings),
                               self.mean(self.get_ndcg(dcg5, idcg5)))
