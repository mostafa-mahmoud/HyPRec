#!/usr/bin/env python
"""
This module provides the functionalities of content-based analysis of the tests.
"""
import numpy
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender


class ContentBased(AbstractRecommender):
    """
    An abstract class that will take the parsed data, and returns a distribution of the content-based information.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True):
        """
        Constructor of ContentBased processor.

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: An evaluator of recommender and holder of input.
        :param dict config: A dictionary of the hyperparameters.
        :param dict options: A dictionary of the run options.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the output matrices.
        """
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.ratings
        self.abstracts_preprocessor = evaluator.abstracts_preprocessor
        self.n_users = self.ratings.shape[0]
        self.n_items = self.abstracts_preprocessor.get_num_items()
        self.set_hyperparameters(hyperparameters)
        self.set_options(options)
        # setting flags
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._verbose = verbose

    @overrides
    def set_options(self, options):
        """
        Set the options of the recommender. Namely n_iterations.

        :param dict options: A dictionary of the options.
        """
        self.n_iter = options['n_iterations']
        self.options = options

    @overrides
    def train(self):
        """
        Train the content-based.
        """
        self.document_distribution = numpy.random.random((self.n_items, self.n_factors))

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        Set the  of the algorithm. Namely n_factors.

        :param dict hyperparameters: A dictionary of the hyperparameters.
        """
        self.n_factors = hyperparameters['n_factors']
        self.hyperparameters = hyperparameters

    def get_document_topic_distribution(self):
        """
        Get the matrix of document X topics distribution.

        :returns: A matrix of documents X topics distribution.
        :rtype: ndarray
        """
        return self.document_distribution

    @overrides
    def get_predictions(self):
        """
        Get the expected ratings between users and items.

        :returns: A matrix of users X documents
        :rtype: ndarray
        """
        # The matrix V * VT is a (cosine) similarity matrix, where V is the row-normalized
        # latent document matrix, this matrix is big, so we avoid having it in inline computations
        # by changing the multiplication order
        # predicted_rating[u,i] = sum[j]{R[u,j] Vj * Vi} / sum[j]{Vj * Vi}
        #                       = sum[j]{R[u,j] * cos(i, j)} / sum[j]{cos(i, j)}
        V = self.document_distribution.copy()
        for item in range(V.shape[0]):
            mag = numpy.sqrt(V[item].dot(V[item]))
            if mag > 1e-6:
                V[item] /= mag
        weighted_ratings = self.ratings.dot(V).dot(V.T)
        weights = V.dot(V.T.dot(numpy.ones((V.shape[0],))))
        self.predictions = weighted_ratings / weights  # Divisions by zero are handled.
        self.predictions[~numpy.isfinite(self.predictions)] = 0.0
        return self.predictions
