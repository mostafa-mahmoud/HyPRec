#!/usr/bin/env python3
"""
A module to run different recommenders.
"""
import numpy
from lib.evaluator import Evaluator
from lib.LDA import LDARecommender
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration


class RunnableRecommenders(object):
    """
    A class that is used to run recommenders.
    """
    def __init__(self, use_database=True, config=None):
        """
        Setup the data and configuration for the recommenders.
        """
        if use_database:
            self.abstracts = DataParser.get_abstracts().values()
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
            self.documents, self.users = self.ratings.shape
        else:
            self.documents, self.users = 8, 10
            self.abstracts = ({'1': 'hell world berlin dna evolution', '2': 'freiburg is green',
                               '3': 'the best dna is the dna of dinasours', '4': 'truth is absolute',
                               '5': 'berlin is not that green', '6': 'truth manifests',
                               '7': 'plato said truth is beautiful', '8': 'freiburg has dna'}).values()
            self.ratings = [[int(not bool((article + user) % 3)) for article in range(self.documents)]
                            for user in range(self.users)]

        self.evaluator = Evaluator(self.ratings, self.abstracts)
        if not config:
            self.config = RecommenderConfiguration()
        else:
            self.config = config
        self.hyperparameters = self.config.get_hyperparameters()
        self.n_iterations = self.config.get_options()['n_iterations']

    def run_lda(self):
        """
        Run LDA recommender.
        """
        lda_recommender = LDARecommender(self.abstracts, self.evaluator, self.hyperparameters)
        lda_recommender.train(self.n_iterations)
        return lda_recommender.get_word_distribution()


if __name__ == '__main__':
    runnable = RunnableRecommenders(False)
    print(runnable.run_lda())
