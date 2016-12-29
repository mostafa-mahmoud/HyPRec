#!/usr/bin/env python
"""
This is a module that contains the main class and functionalities of the recommender systems.
"""
import numpy
from lib.content_based import ContentBased
from lib.evaluator import Evaluator
from lib.LDA import LDARecommender
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration


class RecommenderSystem(object):
    """
    A class that will combine the content-based and collaborative-filtering,
    in order to provide the main functionalities of recommendations.
    """
    def __init__(self):
        """
        Constructor of the RecommenderSystem.
        """
        DataParser.process()
        self.ratings = DataParser.get_ratings_matrix()
        # TODO: split abstracts
        self.abstracts = DataParser.get_abstracts().values()
        self.config = RecommenderConfiguration()
        self.n_factors = self.config.get_hyperparameters()['n_factors']
        self.n_iterations = self.config.get_options()['n_iterations']
        self.content_based = ContentBased(self.abstracts, self.n_factors, self.n_iterations)
        if self.config.get_content_based() == 'LDA':
            self.content_based = LDARecommender(self.abstracts, self.n_factors, self.n_iterations)
        elif self.config.get_content_based() == 'LDA2Vec':
            raise NotImplemented('LDA2Vec is not yet implemented.')
        else:
            raise NameError("Not a valid content based " + self.config.get_content_based())
        self.hyperparameters = self.config.get_hyperparameters()
        if self.config.get_collaborative_filtering() == 'ALS':
        # self.collaborative_filtering = CollaborativeFiltering(ratings, self.n_factors,
        #                                                       self.hyperparameters['collaborative-filtering-lambda'])
            pass
        else:
            raise NameError("Not a valid collaborative filtering " + self.config.get_collaborative_filtering())
        if self.config.get_error_metric() == 'RMS':
            # TODO: initialize with abstracts
            self.evaluator = Evaluator(self.ratings)
        else:
            raise NameError("Not a valid error metric " + self.config.get_error_metric())

    def process(self):
        """
        Process an iteration of the algorithm on the given data.
        """
        self.content_based.train()
        theta = self.content_based.get_word_distribution()
        # TODO: Use collaborative filtering and evaluator
        # u, v = self.collaborative_filtering.train(theta)
        error = self.evaluator.get_rmse(theta)
        return error

    def recommend_items(self, user_id, num_recommendations=10):
        """
        Get recommendations for a user.
        @param(int) user_id: The id of the user.
        @param(int) num_recommendations: The number of recommended items.
        @returns(list) a list of the best recommendations for a given user_id.
        """
        pass
