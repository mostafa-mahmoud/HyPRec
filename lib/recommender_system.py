#!/usr/bin/env python
"""
This is a module that contains the main class and functionalities of the recommender systems.
"""
import numpy
from lib.collaborative_filtering import CollaborativeFiltering
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
        self.ratings = numpy.array(DataParser.get_ratings_matrix())
        self.abstracts = DataParser.get_abstracts().values()
        self.config = RecommenderConfiguration()
        self.hyperparameters = self.config.get_hyperparameters()
        self.n_iterations = self.config.get_options()['n_iterations']
        if self.config.get_error_metric() == 'RMS':
            self.evaluator = Evaluator(self.ratings, self.abstracts)
        else:
            raise NameError("Not a valid error metric " + self.config.get_error_metric())

        self.content_based = ContentBased(self.abstracts, self.evaluator, self.hyperparameters)
        if self.config.get_content_based() == 'LDA':
            self.content_based = LDARecommender(self.abstracts, self.evaluator, self.hyperparameters)
        elif self.config.get_content_based() == 'LDA2Vec':
            raise NotImplementedError('LDA2Vec is not yet implemented.')
        else:
            raise NameError("Not a valid content based " + self.config.get_content_based())

        if self.config.get_collaborative_filtering() == 'ALS':
            self.collaborative_filtering = CollaborativeFiltering(self.ratings, self.evaluator, self.hyperparameters)
        else:
            raise NameError("Not a valid collaborative filtering " + self.config.get_collaborative_filtering())

    def train(self):
        """
        Train the recommender on the given data.
        @returns (float) The RMS error of the predictions.
        """
        self.content_based.train(self.n_iterations)
        theta = self.content_based.get_word_distribution()
        self.collaborative_filtering.train(theta, self.n_iterations)
        error = self.evaluator.get_rmse(self.collaborative_filtering.get_predictions())
        return error

    def recommend_items(self, user_id, num_recommendations=10):
        """
        Get recommendations for a user.
        @param(int) user_id: The id of the user.
        @param(int) num_recommendations: The number of recommended items.
        @returns(list) a list of the best recommendations for a given user_id.
        """
        pass
