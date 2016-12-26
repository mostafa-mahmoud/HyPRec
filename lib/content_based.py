#!/usr/bin/env python
"""
This module provides the functionalities of content-based analysis of the tests.
"""
import numpy


class ContentBased(object):
    """
    An abstract class that will take the parsed data, and returns a distribution of the content-based information.
    """
    def __init__(self, ratings, n_factors):
        """
        Constructor of ContentBased processor.
        """
        self.n_factors = n_factors
        self.n_items = ratings.shape[0]
        self.ratings = ratings

    def train(self, num_iterations=10):
        """
        Train the content-based.
        @param num_iterations(int): number of iterations of the training.
        """
        self.distribution = numpy.random.random((self.n_items, self.n_factors))

    def get_word_distribution(self):
        """
        @returns A matrix of the words x topics distribution.
        """
        return self.distribution
