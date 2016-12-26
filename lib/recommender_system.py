#!/usr/bin/env python
"""
This is a module that contains the main class and functionalities of the recommender systems.
"""
import numpy
from content_based import ContentBased
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration


class RecommenderSystem(object):
    """
    A class that will combine the content-based and collaborative-filtering,
    in order to provide the main functionalities of recommendations.
    """
    def __init__(self, n_factors=4):
        """
        Constructor of the RecommenderSystem.
        """
        # DataParser.process()
        ratings = numpy.matrix(DataParser.get_ratings_matrix())
        self.config = RecommenderConfiguration()
        if self.config.get_content_based() == 'LDA':
            pass
        if self.config.get_content_based() == 'LDA2Vec':
            pass
        # TODO: get vocabulary matrix for content based.
        self.content_based = ContentBased(ratings, n_factors)
        self.hyperparameters = self.config.get_hyperparameters()
        # TODO: write the evaluator class, and hyperparameters for the algorithms.
        # self.collaborative_filtering = CollaborativeFiltering(ratings, n_factors,
        #                                                       self.hyperparameters['collaborative-filtering-lambda'])
        # self.evaluator = Evaluator(self.config.get_error_metric())

    def process(self):
        """
        Process an iteration of the algorithm on the given data.
        """
        self.content_based.train()
        theta = self.content_based.get_word_distribution()
        # TODO: Use collaborative filtering and evaluator
        # u, v = self.collaborative_filtering.train(theta)
        # error = self.evaluator.process(u, v)
        # return error

    def recommend_items(self, user_id, num_recommendations=10):
        """
        @return a list of the best recommendations for a given user_id.
        """
        pass
