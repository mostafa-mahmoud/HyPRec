#!/usr/bin/env python
import numpy
import unittest
from util.top_recommendations import TopRecommendations


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.sizes = [10, 100, 1000]
        self.top_recommendations_array = []
        for size in self.sizes:
            self.top_recommendations_array.append(TopRecommendations(size))


class TestTopRecommendations(TestcaseBase):
    def runTest(self):
        for size, top_recommendation in zip(self.sizes, self.top_recommendations_array):
            self.recommendationsDoesNotExceedSize(top_recommendation, size)
            self.recommendationDoesNotExceedInsertedItems(top_recommendation, size)
            self.recommendationsIsSorted(top_recommendation)

    def recommendationsIsSorted(self, top_recommendations):
        rec_arr = top_recommendations.get_values()
        self.assertTrue(all(rec_arr[i] <= rec_arr[i+1] for i in range(len(rec_arr) - 1)))

    def recommendationsDoesNotExceedSize(self, top_recommendations, size):
        for i in range(size + 20):
            top_recommendations.insert(numpy.random.random(), int(numpy.random.random() * size))
        recommendations_length = len(top_recommendations.get_values())
        indices_length = len(top_recommendations.get_indices())
        returned_size = top_recommendations.get_recommendations_count()
        self.assertTrue(recommendations_length == indices_length == size == returned_size)

    def recommendationDoesNotExceedInsertedItems(self, top_recommendations, size):
        items_count = size - int(size * numpy.random.random())
        for i in range(items_count):
            top_recommendations.insert(numpy.random.random(), int(numpy.random.random() * size))
        recommendations_length = len(top_recommendations.get_values())
        indices_length = len(top_recommendations.get_indices())
        returned_size = top_recommendations.get_recommendations_count()
        self.assertTrue(recommendations_length == indices_length == size == returned_size)
