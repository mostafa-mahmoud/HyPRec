#!/usr/bin/env python3
"""
A module to run different recommenders.
"""
import sys
import itertools
import numpy
from optparse import OptionParser
from lib.evaluator import Evaluator
from lib.collaborative_filtering import CollaborativeFiltering
from lib.grid_search import GridSearch
from lib.LDA import LDARecommender
from lib.LDA2Vec import LDA2VecRecommender
from lib.recommender_system import RecommenderSystem
from util.abstracts_preprocessor import AbstractsPreprocessor
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
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
            self.documents, self.users = self.ratings.shape
            self.abstracts_preprocessor = AbstractsPreprocessor(DataParser.get_abstracts(),
                                                                *DataParser.get_word_distribution())
        else:
            abstracts = {1: 'hell world berlin dna evolution', 2: 'freiburg is green',
                         3: 'the best dna is the dna of dinasours', 4: 'truth is absolute',
                         5: 'berlin is not that green', 6: 'truth manifests itself',
                         7: 'plato said truth is beautiful', 8: 'freiburg has dna'}

            vocab = set(itertools.chain(*list(map(lambda ab: ab.split(' '), abstracts.values()))))
            w2i = dict(zip(vocab, range(len(vocab))))
            word_to_count = [(w2i[word], sum(abstract.split(' ').count(word)
                                             for doc_id, abstract in abstracts.items())) for word in vocab]
            article_to_word = list(set([(doc_id, w2i[word])
                                        for doc_id, abstract in abstracts.items() for word in abstract.split(' ')]))
            article_to_word_to_count = list(set([(doc_id, w2i[word], abstract.count(word))
                                                 for doc_id, abstract in abstracts.items()
                                                 for word in abstract.split(' ')]))
            self.abstracts_preprocessor = AbstractsPreprocessor(abstracts, word_to_count,
                                                                article_to_word, article_to_word_to_count)
            self.documents, self.users = 8, 10
            self.ratings = numpy.array([[int(not bool((article + user) % 3))
                                         for article in range(self.documents)]
                                        for user in range(self.users)])

        self.evaluator = Evaluator(self.ratings, self.abstracts_preprocessor)
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
        lda_recommender = LDARecommender(self.abstracts_preprocessor, self.evaluator,
                                         self.hyperparameters, verbose=True)
        lda_recommender.train(self.n_iterations)
        print(lda_recommender.get_document_topic_distribution().shape)
        return lda_recommender.get_document_topic_distribution()

    def run_lda2vec(self):
        """
        Runs LDA2Vec recommender.
        """
        lda2vec_recommender = LDA2VecRecommender(self.abstracts_preprocessor, self.evaluator,
                                                 self.hyperparameters, verbose=True)
        lda2vec_recommender.train(self.n_iterations)
        print(lda2vec_recommender.get_document_topic_distribution().shape)
        return lda2vec_recommender.get_document_topic_distribution()

    def run_collaborative(self):
        """
        Runs collaborative filtering
        """
        ALS = CollaborativeFiltering(self.ratings, self.evaluator, self.hyperparameters, verbose=True)
        train, test = ALS.naive_split()
        ALS.train()
        print(ALS.evaluator.calculate_recall(ALS.ratings, ALS.rounded_predictions()))
        return ALS.evaluator.recall_at_x(50, ALS.get_predictions())

    def run_grid_search(self):
        """
        runs grid search
        """
        hyperparameters = {
            '_lambda': [0.00001, 0.01, 0.1, 0.5, 10, 100],
            'n_factors': [20, 40, 100, 200, 300]
        }
        print(type(self.ratings))
        ALS = CollaborativeFiltering(self.ratings, self.evaluator, self.hyperparameters, verbose=True)
        GS = GridSearch(ALS, hyperparameters)
        best_params = GS.train()
        return best_params

    def run_recommender(self):
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor,
                                        ratings=self.ratings, verbose=True)
        error = recommender.train()
        print(recommender.content_based.get_document_topic_distribution().shape)
        return error

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--use-database", dest="db", action='store_true',
                      help="use database to run the recommender", metavar="DB")
    parser.add_option("-a", "--all", dest="all", action='store_true',
                      help="run every method", metavar="ALL")
    options, args = parser.parse_args()
    use_database = options.db is not None
    all = options.all is not None
    runnable = RunnableRecommenders(use_database)
    if all is True:
        print(runnable.run_recommender())
        print(runnable.run_collaborative())
        print(runnable.run_grid_search())
        print(runnable.run_lda())
        print(runnable.run_lda2vec())
        sys.exit(0)
    found_runnable = False
    for arg in args:
        if arg == 'recommender':
            print(runnable.run_recommender())
            found_runnable = True
        elif arg == 'collaborative':
            print(runnable.run_collaborative())
            found_runnable = True
        elif arg == 'grid_search':
            print(runnable.run_grid_search())
            found_runnable = True
        elif arg == 'lda':
            print(runnable.run_lda())
            found_runnable = True
        elif arg == 'lda2vec':
            print(runnable.run_lda2vec())
            found_runnable = True
        else:
            print("'%s' option is not valid, please use one of \
                  ['recommender', 'collaborative', 'grid_search', 'lda', 'lda2vec']" % arg)
    if found_runnable is False:
        print("Didn't find any valid option, running recommender instead.")
        print(runnable.run_recommender())
