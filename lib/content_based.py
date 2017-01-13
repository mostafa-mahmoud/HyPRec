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
    def __init__(self, abstracts_preprocessor, evaluator, config, verbose=False):
        """
        Constructor of ContentBased processor.

        :param AbstractsPreprocessor abstracts_preprocessor: Abstracts preprocessor
        :param Evaluator evaluator: An evaluator object.
        :param dict config: A dictionary of the hyperparameters.
        :param boolean verbose: A flag for printing while computing.
        """
        self.set_config(config)
        self.abstracts_preprocessor = abstracts_preprocessor
        self.n_items = self.abstracts_preprocessor.get_num_items()
        self.evaluator = evaluator
        self._v = verbose

    def train(self, n_iter=5):
        """
        Train the content-based.

        :param int n_iter: The number of iterations of training the model.
        """
        self.document_distribution = numpy.random.random((self.n_items, self.n_factors))
        for _ in range(n_iter):
            pass

    def set_config(self, config):
        """
        Set the hyperparamenters of the algorithm.

        :param dict config: A dictionary of the hyperparameters.
        """
        self.n_factors = config['n_factors']

    def get_document_topic_distribution(self):
        """
        Get the matrix of document X topics distribution.

        :returns: A matrix of documents X topics distribution.
        :rtype: ndarray
        """
        return self.document_distribution
