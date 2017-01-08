#!/usr/bin/env python
"""
Module that provides the main functionalities of collaborative filtering.
"""

from numpy.linalg import solve
import numpy
from lib.abstract_recommender import AbstractRecommender


class CollaborativeFiltering(AbstractRecommender):
    """
    A class that takes in the rating matrix and outputs user and item
    representation in latent space.
    """
    def __init__(self, ratings, evaluator, config, verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a ratings matrix which is ~ user x item

        :param ndarray ratings:
            A matrix containing the ratings 1 indicates user has the document in his library
            0 indicates otherwise.
        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors
        :param Evaluator evaluator: object that evaluates the recommender
        :param boolean verbose: A flag if True, tracing will be printed
        """
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.set_config(config)
        self.evaluator = evaluator
        self._v = verbose

    def set_config(self, config):
        """
        The function sets the config of the uv_decomposition algorithm

        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = config['n_factors']
        self._lambda = config['_lambda']

    def split(self, test_percentage=0.2):
        """
        Split the ratings into test and train data.

        :param float test_percentage: The ratio of the testing data from all the data.
        :returns: a tuple of train and test data.
        :rtype: tuple
        """
        test = numpy.zeros(self.ratings.shape)
        train = self.ratings.copy()
        # TODO split in a more intelligent way
        for user in range(self.ratings.shape[0]):
            non_zeros = self.ratings[user, :].nonzero()[0]
            test_ratings = numpy.random.choice(non_zeros,
                                               size=int(test_percentage * len(non_zeros)))
            train[user, test_ratings] = 0.
            test[user, test_ratings] = self.ratings[user, test_ratings]
        assert(numpy.all((train * test) == 0))
        self.ratings = train
        return train, test

    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """
        The function computes only one step in the ALS algorithm

        :param ndarray latent_vectors: the vector to be optimized
        :param ndarray fixed_vecs: the vector to be fixed
        :param ndarray ratings: ratings that will be used to optimize latent * fixed
        :param float _lambda: reguralization parameter
        :param str type: either user or item.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = numpy.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = numpy.eye(XTX.shape[0]) * _lambda
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, item_vecs=None, n_iter=15):
        """
        Train model for n_iter iterations from scratch.

        :param int n_iter: number of iterations
        """
        self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
        if item_vecs is None:
            self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
        else:
            self.item_vecs = item_vecs
        self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be called multiple times for further training.

        :param int n_iter: number of iterations
        """
        ctr = 1
        while ctr <= n_iter:
            if self._v:
                print('\tcurrent iteration: {}'.format(ctr))
                print('Error %f' % self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.ratings))
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.ratings, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.ratings, self._lambda, type='item')
            ctr += 1

    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: predictions
        :rtype: ndarray
        """
        return self.user_vecs.dot(self.item_vecs.T)

    def predict(self, user, item):
        """
        Single user and item prediction.

        :returns: prediction score
        :rtype: float
        """
        return self.user_vecs[user, :].dot(self.item_vecs[item, :].T)

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
