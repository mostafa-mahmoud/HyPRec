#!/usr/bin/env python
"""
Module that trains a linear regression model to combine
content based and collaborative recommenders.
"""
import numpy
from sklearn import linear_model


class LinearRegression(object):
    """
    Linear regression to combine the results of two matrices.
    """
    def __init__(self, train_labels, test_labels, item_based_ratings, collaborative_ratings):
        """
        Apply linear regression between two different methods to predict final collaborative_ratings

        :param ndarray train_labels: Training data.
        :param ndarray test_labels: Test data.
        :param ndarray item_based_ratings: Ratings produced by item based recommender.
        :param ndarray collaborative_ratings: Ratings produced by collaborative recommender
        """
        self.item_based_ratings = item_based_ratings
        self.collaborative_ratings = collaborative_ratings
        self.item_based_ratings_shape = item_based_ratings.shape
        self.collaborative_ratings_shape = collaborative_ratings.shape
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.flatten_matrices()
        self.train_data = numpy.vstack((self.flat_item_based_ratings, self.flat_collaborative_ratings)).T
        self.regression_coef1 = 0
        self.regression_coef1 = 0


    def flatten_matrices(self):
        """
        Method converts all 2d ndarray to 1d array to be used for linear regression
        """
        self.flat_item_based_ratings = self.flatten_matrix(self.item_based_ratings)
        self.flat_collaborative_ratings = self.flatten_matrix(self.collaborative_ratings)
        self.flat_train_labels = self.flatten_matrix(self.train_labels)
        self.flat_test_labels = self.flatten_matrix(self.test_labels)

    def flatten_matrix(self, matrix):
        """
        Method converts a matrix to a 1d array

        :param ndarray matrix: The matrix to be converted.
        :returns: flattened list
        :rtype: float[]
        """
        return matrix.flatten()

    def unflatten(self, matrix, shape):
        """
        Methods converts 1d array to a 2d array given a shape.

        :param float[] matrix: list to be converted.
        :param tuple(int) shape: Shape of the new matrix.

        :returns: 2D matrix
        :rtype: ndarray
        """
        return matrix.reshape(shape)

    def train(self):
        """
        Method trains a liner regression model

        :returns: adjusted predictions matrix.
        :rtype: ndarray
        """
        regr_model = linear_model.LinearRegression()
        regr_model.fit(self.train_data, self.flat_train_labels)
        weighted_item_based_ratings = regr_model.coef_[0] * self.item_based_ratings
        weighted_collaborative_ratings = regr_model.coef_[1] * self.collaborative_ratings
        self.regression_coef1 = regr_model.coef_[0]
        self.regression_coef2 = regr_model.coef_[1]
        return weighted_collaborative_ratings + weighted_item_based_ratings
