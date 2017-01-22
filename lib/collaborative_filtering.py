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
    def __init__(self, initializer, n_iter, ratings, evaluator, config,
                 verbose=False, load_matrices=True, dump=True, train_more=True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a ratings matrix which is ~ user x item

        :param ModelInitializer initializer: A model initializer.
        :param int n_iter: Number of iterations.
        :param ndarray ratings:
            A matrix containing the ratings 1 indicates user has the document in his library
            0 indicates otherwise.
        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors
        :param Evaluator evaluator: object that evaluates the recommender
        :param boolean verbose: A flag if True, tracing will be printed
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump: A flag for saving the matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        """
        self.dump = dump
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.set_config(config)
        self.evaluator = evaluator
        self.n_iter = n_iter
        self.initializer = initializer
        self.load_matrices = load_matrices
        self._v = verbose
        self._train_more = train_more
        self.naive_split()

    def set_iterations(self, n_iter):
        self.n_iter = n_iter

    def set_config(self, config):
        """
        The function sets the config of the uv_decomposition algorithm

        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = config['n_factors']
        self._lambda = config['_lambda']
        self.config = config

    def naive_split(self, test_percentage=0.2):
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
        self.train_data = train
        self.test = test
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

    def train(self, item_vecs=None):
        """
        Train model for n_iter iterations from scratch.

        """
        matrices_found = False
        if self.load_matrices is False:
            self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
            if item_vecs is None:
                self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
            else:
                self.item_vecs = item_vecs
        else:
            users_found, self.user_vecs = self.initializer.load_matrix(self.config,
                                                                       'user_vecs', (self.n_users, self.n_factors))
            if self._v and users_found:
                print("User distributions files were found.")
            if item_vecs is None:
                items_found, self.item_vecs = self.initializer.load_matrix(self.config, 'item_vecs',
                                                                           (self.n_items, self.n_factors))
                if self._v and items_found:
                    print("Document distributions files were found.")
            else:
                items_found = True
                self.item_vecs = item_vecs
            matrices_found = users_found and items_found
        if not matrices_found:
            if self._v and self.load_matrices:
                print("User and Document distributions files were not found, will train collaborative.")
            self.partial_train()
        else:
            if self._train_more:
                if self._v and self.load_matrices:
                    print("User and Document distributions files found, will train model further.")
                self.partial_train()
            else:
                if self._v and self.load_matrices:
                    print("User and Document distributions files found, will not train the model further.")

            if self._v:
                print("User and Document distributions files found, will train model further.")
        if self.dump:
            self.initializer.set_config(self.config, self.n_iter)
            self.initializer.save_matrix(self.user_vecs, 'user_vecs')
            self.initializer.save_matrix(self.item_vecs, 'item_vecs')
        if self._v:
            predictions = self.get_predictions()
            rounded_predictions = self.rounded_predictions()
            self.evaluator.load_top_recommendations(200, predictions)
            train_recall = self.evaluator.calculate_recall(self.train_data, rounded_predictions)
            test_recall = self.evaluator.calculate_recall(self.test, rounded_predictions)
            recall_at_x = self.evaluator.recall_at_x(200, predictions)
            recommendations = sum(sum(rounded_predictions))
            likes = sum(sum(self.ratings))
            ratio = recommendations / likes
            mrr_at_five = self.evaluator.calculate_mrr(5, predictions)
            ndcg_at_five = self.evaluator.calculate_ndcg(5, predictions)
            mrr_at_ten = self.evaluator.calculate_mrr(10, predictions)
            ndcg_at_ten = self.evaluator.calculate_ndcg(10, predictions)
            print('Final Error %f, train recall %f, test recall %f, recall at 200 %f, ratio %f, mrr @5 %f, ndcg @5 %f, mrr @10 %f,\
                   ndcg @10 %f' % (self.evaluator.get_rmse(predictions, self.ratings), train_recall,
                                   test_recall, recall_at_x, ratio, mrr_at_five, ndcg_at_five,
                                   mrr_at_ten, ndcg_at_ten))

    def partial_train(self):
        """
        Train model for n_iter iterations. Can be called multiple times for further training.

        """
        ctr = 1
        while ctr <= self.n_iter:
            if self._v:
                print('\tcurrent iteration: {}'.format(ctr))
                print('Error %f' % self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.ratings))
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')
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
