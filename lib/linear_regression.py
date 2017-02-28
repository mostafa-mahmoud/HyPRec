#!/usr/bin/env python
"""
Module that trains a linear regression model to combine
content based and collaborative recommenders.
"""
import numpy
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold

class LinearRegression(object):

    def __init__(self, train_labels, test_labels, item_based_ratings, collaborative_ratings):
        self.item_based_ratings = item_based_ratings
        self.collaborative_ratings = collaborative_ratings
        self.item_based_ratings_shape = item_based_ratings.shape
        self.collaborative_ratings_shape = collaborative_ratings.shape
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.flatten_matrices()
        self.train_data = numpy.vstack((self.flat_item_based_ratings, self.flat_collaborative_ratings)).T

    def flatten_matrices(self):
        self.flat_item_based_ratings = self.flatten_matrix(self.item_based_ratings)
        self.flat_collaborative_ratings = self.flatten_matrix(self.collaborative_ratings)
        self.flat_train_labels = self.flatten_matrix(self.train_labels)
        self.flat_test_labels = self.flatten_matrix(self.test_labels)

    def flatten_matrix(self, matrix):
        return matrix.flatten()

    def unflatten(self, matrix, shape):
        return matrix.reshape(shape)

    def train(self):
        regr_model = linear_model.LinearRegression()
        regr_model.fit(self.train_data, self.flat_train_labels)
        weighted_item_based_ratings = regr_model.coef_[0] * self.item_based_ratings
        weighted_collaborative_ratings = regr_model.coef_[1] * self.collaborative_ratings
        return weighted_collaborative_ratings + weighted_item_based_ratings



