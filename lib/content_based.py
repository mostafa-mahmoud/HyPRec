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
        self.ratings = evaluator.get_ratings()
        self.abstracts_preprocessor = evaluator.get_abstracts_preprocessor()
        self.n_users, self.n_items = self.ratings.shape
        assert self.n_items == self.abstracts_preprocessor.get_num_items()
        # setting flags
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._verbose = verbose
        self.set_hyperparameters(hyperparameters)
        self.set_options(options)

    @overrides
    def train_k_fold(self):
        """
        Trains k folds of the content based.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        self.train_one_fold()
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_train_indices,
                                                                      self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            all_errors.append(self.get_evaluation_report())
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self):
        """
        Train one fold for n_iter iterations from scratch.
        """
        self.document_distribution = None

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        Set the  of the algorithm. Namely n_factors.

        :param dict hyperparameters: A dictionary of the hyperparameters.
        """
        self.n_factors = hyperparameters['n_factors']
        self.predictions = None
        self.hyperparameters = hyperparameters.copy()

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
        if self.predictions is not None:
            return self.predictions
        # The matrix V * VT is a (cosine) similarity matrix, where V is the row-normalized
        # latent document matrix, this matrix is big, so we avoid having it in inline computations
        # by changing the multiplication order
        # predicted_rating[u,i] = sum[j]{R[u,j] Vj * Vi} / sum[j]{Vj * Vi}
        #                       = sum[j]{R[u,j] * cos(i, j)} / sum[j]{cos(i, j)}
        if self.document_distribution is None:
            V = numpy.random.random((self.n_items, self.n_factors))
        else:
            V = self.document_distribution.copy()
        for item in range(V.shape[0]):
            mean = numpy.mean(V[item])
            V[item] -= mean
            item_norm = numpy.sqrt(V[item].dot(V[item]))
            if item_norm > 1e-6:
                V[item] /= item_norm
        weighted_ratings = self.train_data.dot(V).dot(V.T)
        weights = V.dot(V.T.dot(numpy.ones((V.shape[0],))))
        self.predictions = weighted_ratings / weights  # Divisions by zero are handled.
        self.predictions[~numpy.isfinite(self.predictions)] = 0.0
        return self.predictions
