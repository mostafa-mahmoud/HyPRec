#!/usr/bin/env python
"""
Module that provides the main functionalities of collaborative filtering.
"""

from numpy.linalg import solve
import numpy
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util.data_parser import DataParser
from lib.evaluator import Evaluator
from lib.abstract_recommender import AbstractRecommender
import time
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation as cv

class CollaborativeFiltering(AbstractRecommender):
    """
    A class that takes in the rating matrix and outputs user and item
    representation in latent space.
    """
    def __init__(self, 
                 ratings,
                 evaluator,
                 config,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x document matrix with corresponding ratings.
            1 indicates user has the document in his library
            0 indicates otherwise.
        
        n_factors : (int)
            number of latent factors used. Must be the same
            as the one used in the LDA.
        
        _lambda : (float)
            Regularization term to avoid overfitting
        
        verbose : (bool)
            Intermediate tracing will be printed if the variable
            is set to True
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = config['n_factors']
        self._lambda = config['_lambda']
        self.evaluator = evaluator
        self._v = verbose

    def process(self):
        """
        Start processing the data.
        """
        pass

    def set_config(self, config):
        self.n_factors = config['n_factors']
        self._lambda = config['_lambda']

    def split(self):
        test = numpy.zeros(self.ratings.shape)
        train = self.ratings.copy()
        for user in range(self.ratings.shape[0]):
            test_ratings = numpy.random.choice(self.ratings[user, :].nonzero()[0], 
                                        size=10)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = self.ratings[user, test_ratings]
        
        assert(numpy.all((train * test) == 0)) 
        self.ratings = train
        return train, test

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
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

    def train(self, n_iter=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
        self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
        
        self.partial_train(n_iter)
        print("sum")
        print(sum(sum(self.ratings)))
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if self._v:
                print('\tcurrent iteration: {}'.format(ctr))
                print('Error %f' % self.get_mse(self.user_vecs.dot(self.item_vecs.T), self.ratings))
            self.user_vecs = self.als_step(self.user_vecs, 
                                           self.item_vecs, 
                                           self.ratings, 
                                           self._lambda, 
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs, 
                                           self.user_vecs, 
                                           self.ratings, 
                                           self._lambda, 
                                           type='item')
            ctr += 1
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        return self.user_vecs.dot(self.item_vecs.T)

    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)
            predictions = self.predict_all()
            self.train_mse += [self.get_mse(predictions, self.ratings)]
            self.test_mse += [self.get_mse(predictions, test)]
            if self._v:
                print('Train mse: ' + str(self.train_mse[-1]))
                print('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_ite

    def get_mse(self, pred, actual):
        return self.evaluator.get_rmse(pred, actual)

    def get_predictions(self):
        return self.predict_all()

    def get_ratings(self):
        return self.ratings

if __name__ == "__main__":

    R = numpy.array(DataParser.get_ratings_matrix())
    m,n = R.shape
    print("Initial Mean %f Max %f Min %f" % (R.mean(), R.max(), R.min()))
    evaluator = Evaluator(R)
    ALS = CollaborativeFiltering(R, evaluator, {'n_factors': 200, '_lambda': 0.1}, True)
    train, test = ALS.split()
    ALS.train()