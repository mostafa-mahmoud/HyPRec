#!/usr/bin/env python
"""
This module provides the functionalities of content-based analysis of the tests.
"""
import numpy
from lib.abstract_recommender import AbstractRecommender


class ContentBased(AbstractRecommender):
    """
    An abstract class that will take the parsed data, and returns a distribution of the content-based information.
    """
    def __init__(self, abstracts, evaluator, config, verbose=False):
        """
        Constructor of ContentBased processor.
        """
        self.set_config(config)
        self.n_items = len(abstracts)
        self.abstracts = abstracts
        self.evaluator = evaluator
        self._v = verbose

    def train(self, n_iter=5):
        """
        Train the content-based.
        """
        self.word_distribution = numpy.random.random((self.n_items, self.n_factors))
        for _ in range(n_iter):
            pass

    def split(self):
        pass

    def set_config(self, config):
        """
        Set the hyperparamenters of the algorithm.
        """
        self.n_factors = config['n_factors']

    def get_word_distribution(self):
        """
        @returns A matrix of the words x topics distribution.
        """
        return self.word_distribution
