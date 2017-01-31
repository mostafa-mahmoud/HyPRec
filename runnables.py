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
from util.model_initializer import ModelInitializer


class RunnableRecommenders(object):
    """
    A class that is used to run recommenders.
    """
    def __init__(self, use_database=True, verbose=True, load_matrices=True, dump=True, train_more=True, config=None):
        """
        Setup the data and configuration for the recommenders.
        """
        if use_database:
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
            self.documents, self.users = self.ratings.shape
            self.abstracts_preprocessor = AbstractsPreprocessor(DataParser.get_abstracts(),
                                                                *DataParser.get_word_distribution())
        else:
            abstracts = {0: 'hell world berlin dna evolution', 1: 'freiburg is green',
                         2: 'the best dna is the dna of dinasours', 3: 'truth is absolute',
                         4: 'berlin is not that green', 5: 'truth manifests itself',
                         6: 'plato said truth is beautiful', 7: 'freiburg has dna'}

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

        self.verbose = verbose
        self.load_matrices = load_matrices
        self.dump = dump
        self.evaluator = Evaluator(self.ratings, self.abstracts_preprocessor)
        self.train_more = train_more
        if not config:
            self.config = RecommenderConfiguration()
        else:
            self.config = config
        self.hyperparameters = self.config.get_hyperparameters()
        self.n_iterations = self.config.get_options()['n_iterations']
        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iterations, self.verbose)

    def run_lda(self):
        """
        Run LDA recommender.
        """
        lda_recommender = LDARecommender(self.initializer, self.abstracts_preprocessor, self.ratings, self.evaluator,
                                         self.hyperparameters, self.verbose, self.load_matrices, self.dump)
        lda_recommender.train(self.n_iterations)
        print(lda_recommender.get_document_topic_distribution().shape)
        return lda_recommender.get_document_topic_distribution()

    def run_lda2vec(self):
        """
        Runs LDA2Vec recommender.
        """
        lda2vec_recommender = LDA2VecRecommender(self.initializer, self.abstracts_preprocessor, self.ratings,
                                                 self.evaluator, self.hyperparameters,
                                                 self.verbose, self.load_matrices, self.dump)
        lda2vec_recommender.train(self.n_iterations)
        print(lda2vec_recommender.get_document_topic_distribution().shape)
        return lda2vec_recommender.get_document_topic_distribution()

    def run_collaborative(self):
        """
        Runs collaborative filtering
        """

        ALS = CollaborativeFiltering(self.initializer, self.n_iterations, self.ratings, self.evaluator,
                                     self.hyperparameters, self.verbose, self.load_matrices, self.dump)
        ALS.train()
        print(ALS.evaluator.calculate_recall(ALS.ratings, ALS.rounded_predictions()))
        return ALS.evaluator.recall_at_x(50, ALS.get_predictions())

    def run_grid_search(self):
        """
        runs grid search
        """
        hyperparameters = {
            '_lambda': [0.00001, 0.01, 0.1, 0.5, 10, 100],
            'n_factors': [100, 200, 300, 400, 500]
        }
        ALS = CollaborativeFiltering(self.initializer, self.n_iterations, self.ratings, self.evaluator,
                                     self.hyperparameters, self.verbose, self.load_matrices, self.dump,
                                     self.train_more)
        GS = GridSearch(ALS, hyperparameters)
        best_params = GS.train()
        return best_params

    def run_recommender(self):
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                        verbose=self.verbose, load_matrices=self.load_matrices, dump=self.dump,
                                        train_more=self.train_more)
        error = recommender.train()
        print(recommender.content_based.get_document_topic_distribution().shape)
        return error


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--use-database", dest="db", action='store_true',
                      help="use database to run the recommender", metavar="DB")
    parser.add_option("-a", "--all", dest="all", action='store_true',
                      help="run every method", metavar="ALL")
    parser.add_option("-s", "--save", dest="dump", action='store_true',
                      help="dump the saved data into files", metavar="DUMP")
    parser.add_option("-l", "--load", dest="load", action='store_true',
                      help="load saved models from files", metavar="LOAD")
    parser.add_option("-v", "--verbose", dest="verbose", action='store_true',
                      help="print update statements during computations", metavar="VERBOSE")
    parser.add_option("-t", "--train_more", dest="train_more", action='store_true',
                      help="train the collaborative filtering more, after loading matrices", metavar="TRAINMORE")
    options, args = parser.parse_args()
    use_database = options.db is not None
    use_all = options.all is not None
    load_matrices = options.load is not None
    verbose = options.verbose is not None
    dump = options.dump is not None
    train_more = options.train_more is not None

    runnable = RunnableRecommenders(use_database, verbose, load_matrices, dump, train_more)
    if use_all is True:
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
