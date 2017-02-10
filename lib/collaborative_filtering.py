#!/usr/bin/env python
"""
Module that provides the main functionalities of collaborative filtering.
"""
import time
import numpy
from numpy.linalg import solve
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender


class CollaborativeFiltering(AbstractRecommender):
    """
    A class that takes in the rating matrix and outputs user and item
    representation in latent space.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True, train_more=True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a ratings matrix which is ~ user x item

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: Evaluator of the recommender and holder of the input data.
        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        :param dict options: Dictionary of the run options, contains n_iterations and k_folds
        :param boolean verbose: A flag if True, tracing will be printed
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        """
        # setting input
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.get_ratings()
        self.n_users, self.n_items = self.ratings.shape
        self.k_folds = None
        self.set_hyperparameters(hyperparameters)
        self.set_options(options)

        # setting flags
        self._verbose = verbose
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._train_more = train_more

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        The function sets the hyperparameters of the uv_decomposition algorithm

        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.hyperparameters = hyperparameters

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

    @overrides
    def train(self, item_vecs=None):
        """
        Train the collaborative filtering.

        :param ndarray item_vecs: optional initalization for the item_vecs matrix.
        """
        if self.splitting_method == 'naive':
            self.train_data, self.test_data = self.evaluator.naive_split()
            return self.train_one_fold(item_vecs)
        else:
            self.fold_train_indices, self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold(item_vecs)

    @overrides
    def train_k_fold(self, item_vecs=None):
        """
        Trains the k folds of collaborative filtering.
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_train_indices,
                                                                      self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            all_errors.append(self.train_one_fold(item_vecs))
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self, item_vecs=None):
        """
        Train model for n_iter iterations from scratch.
        """
        matrices_found = False
        if self._load_matrices is False:
            self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
            if item_vecs is None:
                self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
            else:
                self.item_vecs = item_vecs
        else:
            users_found, self.user_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'user_vecs', (self.n_users, self.n_factors))
            if self._verbose and users_found:
                print("User distributions files were found.")
            if item_vecs is None:
                items_found, self.item_vecs = self.initializer.load_matrix(self.hyperparameters, 'item_vecs',
                                                                           (self.n_items, self.n_factors))
                if self._verbose and items_found:
                    print("Document distributions files were found.")
            else:
                items_found = True
                self.item_vecs = item_vecs
            matrices_found = users_found and items_found
        if not matrices_found:
            if self._verbose and self._load_matrices:
                print("User and Document distributions files were not found, will train collaborative.")
            self.partial_train()
        else:
            if self._train_more:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will train model further.")
                self.partial_train()
            else:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will not train the model further.")

        if self._dump_matrices:
            self.initializer.set_config(self.hyperparameters, self.n_iter)
            self.initializer.save_matrix(self.user_vecs, 'user_vecs')
            self.initializer.save_matrix(self.item_vecs, 'item_vecs')

        return self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.ratings)

    def partial_train(self):
        """
        Train model for n_iter iterations. Can be called multiple times for further training.
        """
        if 'fold' in self.hyperparameters:
            current_fold = self.hyperparameters['fold'] + 1
        else:
            current_fold = 0
        if self._verbose:
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.ratings)
            if current_fold == 0:
                print('E:{epoch:05d} L:{loss:1.4e} T:{tim:.3f}s'.format(**dict(epoch=0, loss=error, tim=0)))
            else:
                print('F:{fold:05d} E:{epoch:05d} L:{loss:1.4e} T:{tim:.3f}s'.format(**dict(fold=current_fold, epoch=0,
                                                                                            loss=error, tim=0)))
        for epoch in range(1, self.n_iter + 1):
            t0 = time.time()
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')
            t1 = time.time()
            if self._verbose:
                error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.ratings)
                if current_fold == 0:
                    print('E:{epoch:05d} L:{loss:1.4e} T:{tim:.3f}s'.format(**dict(epoch=epoch, loss=error,
                                                                                   tim=(t1-t0))))
                else:
                    print('F:{fold:05d} E:{epoch:05d} L:{loss:1.4e} T:{tim:.3f}s'.format(**dict(fold=current_fold,
                                                                                                epoch=epoch,
                                                                                                loss=error,
                                                                                                tim=(t1-t0))))

    @overrides
    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        return self.user_vecs.dot(self.item_vecs.T)

    @overrides
    def predict(self, user, item):
        """
        Single user and item prediction.

        :returns: prediction score
        :rtype: float
        """
        return self.user_vecs[user, :].dot(self.item_vecs[item, :].T)
