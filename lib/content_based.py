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
    def __init__(self, initializer, abstracts_preprocessor, ratings, evaluator, config,
                 verbose=False, load_matrices=True, dump=True):
        """
        Constructor of ContentBased processor.

        :param ModelInitializer initializer: A model initializer.
        :param AbstractsPreprocessor abstracts_preprocessor: Abstracts preprocessor
        :param Ratings ratings: Ratings matrix
        :param Evaluator evaluator: An evaluator object.
        :param dict config: A dictionary of the hyperparameters.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump: A flag for saving the matrices.
        """
        self.set_config(config)
        self.initializer = initializer
        self.ratings = ratings
        self.abstracts_preprocessor = abstracts_preprocessor
        self.n_items = self.abstracts_preprocessor.get_num_items()
        self.evaluator = evaluator
        self.load_matrices = load_matrices
        self.dump = dump
        self._v = verbose

    def train(self, n_iter=5):
        """
        Train the content-based.

        :param int n_iter: The number of iterations of training the model.
        """
        self.document_distribution = numpy.random.random((self.n_items, self.n_factors))
        for _ in range(n_iter):
            pass

    def naive_split(self):
        """
        split the data into train and test data.

        :returns: A tuple of (train_data, test_data)
        :rtype: tuple
        """
        pass

    def set_config(self, config):
        """
        Set the hyperparamenters of the algorithm.

        :param dict config: A dictionary of the hyperparameters.
        """
        self.n_factors = config['n_factors']
        self.config = config

    def get_document_topic_distribution(self):
        """
        Get the matrix of document X topics distribution.

        :returns: A matrix of documents X topics distribution.
        :rtype: ndarray
        """
        return self.document_distribution

    def predicted_ratings(self):
        """
        Get the expected ratings between users and items.

        :returns: A matrix of users X documents
        :rtype: float[][]
        """
        VT = self.document_distribution
        V = self.document_distribution.T
        ratings = self.evaluator.get_ratings()
        mean_ratings = ratings / numpy.sum(ratings, axis=1)
        predicted_ratings = (ratings - mean_ratings).dot(VT).dot(V) / VT.dot(V) + mean_ratings
        return predicted_ratings
