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
    def __init__(self, initializer, abstracts_preprocessor, ratings, evaluator, config, n_iter,
                 verbose=False, load_matrices=True, dump=True):
        """
        Constructor of ContentBased processor.

        :param ModelInitializer initializer: A model initializer.
        :param AbstractsPreprocessor abstracts_preprocessor: Abstracts preprocessor
        :param Ratings ratings: Ratings matrix
        :param Evaluator evaluator: An evaluator object.
        :param dict config: A dictionary of the hyperparameters.
        :param int n_iter: Number of iterations.
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
        self.n_iter = n_iter
        self.load_matrices = load_matrices
        self.dump = dump
        self._v = verbose

    def train(self):
        """
        Train the content-based.

        :param int n_iter: The number of iterations of training the model.
        """
        self.document_distribution = numpy.random.random((self.n_items, self.n_factors))

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

    def get_predictions(self):
        """
        Get the expected ratings between users and items.

        :returns: A matrix of users X documents
        :rtype: float[][]
        """
        VT = self.document_distribution
        V = self.document_distribution.T
        # mean_ratings = (self.ratings.T / numpy.sum(self.ratings, axis=1)).T
        # normalized_ratings = self.ratings - mean_ratings
        # predicted_ratings = normalized_ratings.dot(VT).dot(V) / numpy.sum(VT.dot(V), axis=1) + mean_ratings
        predicted_ratings = self.ratings.dot(VT).dot(V) / numpy.sum(VT.dot(V), axis=1)
        return predicted_ratings

    def get_ratings(self):
        """
        Getter for the ratings

        :returns: Ratings matrix
        :rtype: ndarray
        """
        return self.ratings

    def rounded_predictions(self):
        """
        The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.

        :returns: predictions rounded up matrix
        :rtype: int[][]
        """
        predictions = self.get_predictions()
        n_users = self.ratings.shape[0]
        for user in range(n_users):
            avg = sum(self.ratings[0]) / self.ratings.shape[1]
            low_values_indices = predictions[user, :] < avg
            predictions[user, :] = 1
            predictions[user, low_values_indices] = 0
        return predictions
