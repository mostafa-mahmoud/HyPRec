#!/usr/bin/env python
"""
This is a module that contains the main class and functionalities of the recommender systems.
"""
import numpy
from lib.abstract_recommender import AbstractRecommender
from lib.collaborative_filtering import CollaborativeFiltering
from lib.content_based import ContentBased
from lib.evaluator import Evaluator
from lib.LDA import LDARecommender
from lib.LDA2Vec import LDA2VecRecommender
from util.abstracts_preprocessor import AbstractsPreprocessor
from util.top_recommendations import TopRecommendations
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration
from util.model_initializer import ModelInitializer


class RecommenderSystem(AbstractRecommender):
    """
    A class that will combine the content-based and collaborative-filtering,
    in order to provide the main functionalities of recommendations.
    """
    def __init__(self, initializer=None, abstracts_preprocessor=None, ratings=None,
                 process_parser=False, verbose=False, load_matrices=True, dump=True, train_more=True):
        """
        Constructor of the RecommenderSystem.

        :param list[str] abstracts: List of abstracts; if None, abstracts get queried from the database.
        :param int[][] ratings: Ratings matrix; if None, matrix gets queried from the database.
        :param boolean process_parser: A Flag deceiding process the dataparser.
        :param boolean verbose: A flag deceiding to print progress.
        :param boolean dump: A flag for saving matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        """
        if process_parser:
            DataParser.process()

        if ratings is None:
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
        else:
            self.ratings = ratings

        self.predictions = numpy.zeros(self.ratings.shape)

        if abstracts_preprocessor is None:
            self.abstracts_preprocessor = AbstractsPreprocessor(DataParser.get_abstracts(),
                                                                *DataParser.get_word_distribution())
        else:
            self.abstracts_preprocessor = abstracts_preprocessor

        # Get configurations
        self.config = RecommenderConfiguration()
        self.set_hyperparameters(self.config.get_hyperparameters())
        self.set_options(self.config.get_options())
        
        # Set flags
        self._verbose = verbose
        self._dump_matrices = dump
        self._load_matrices = load_matrices
        self._train_more = train_more
        self.n_iterations = self.config.get_options()['n_iterations']

        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iterations, self._verbose)
        if self.config.get_error_metric() == 'RMS':
            self.evaluator = Evaluator(self.ratings, self.abstracts_preprocessor)
        else:
            raise NameError("Not a valid error metric " + self.config.get_error_metric())

        # Initialize content based.
        self.content_based = ContentBased(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                          self._verbose, self._load_matrices, self._dump_matrices)
        if self.config.get_content_based() == 'LDA':
            self.content_based = LDARecommender(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                                self._verbose, self._load_matrices, self._dump_matrices)
        elif self.config.get_content_based() == 'LDA2Vec':
            self.content_based = LDA2VecRecommender(self.initializer, self.evaluator, self.hyperparameters,
                                                    self.options, self._verbose,
                                                    self._load_matrices, self._dump_matrices)
        else:
            raise NameError("Not a valid content based " + self.config.get_content_based())

        # Initialize collaborative filtering.
        if self.config.get_collaborative_filtering() == 'ALS':
            self.collaborative_filtering = CollaborativeFiltering(self.initializer, self.evaluator,
                                                                  self.hyperparameters, self.options,
                                                                  self._verbose, self._load_matrices,
                                                                  self._dump_matrices, self._train_more)
        else:
            raise NameError("Not a valid collaborative filtering " + self.config.get_collaborative_filtering())

        # Initialize recommender
        if self.config.get_recommender() == 'itembased':
            self.recommender = self.collaborative_filtering
        elif self.config.get_recommender() == 'userbased':
            self.recommender = self.content_based
        else:
            raise NameError("Invalid recommender type " + self.config.get_recommender())


    # @overrides(AbstractRecommender)
    def set_options(self, options):
        """
        Set the options of the recommender. Namely n_iterations and k_folds.
        
        :param dict options: A dictionary of the options.
        """
        if 'n_iterations' in options.keys():
            self.n_iter = options['n_iterations']
        self.options = options

    # @overrides(AbstractRecommender)
    def set_hyperparameters(self, hyperparameters):
        """
        The function sets the hyperparameters of the uv_decomposition algorithm

        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.hyperparameters = hyperparameters
    
    # @overrides(AbstractRecommender)
    def train(self):
        """
        Train the recommender on the given data.

        :returns: The error of the predictions.
        :rtype: float
        """
        if self._verbose:
            print("Training content-based %s..." % self.content_based)
        self.content_based.train()
        theta = self.content_based.get_document_topic_distribution().copy()
        if self._verbose:
            print("Training collaborative-filtering %s..." % self.collaborative_filtering)
        self.collaborative_filtering.train(theta)
        error = self.evaluator.recall_at_x(50, self.collaborative_filtering.get_predictions())
        if self._verbose:
            print("done training...")
        return error

    # @overrides(AbstractRecommender)
    def get_predictions(self):
        """
        Get the predictions matrix
        """
        return self.recommender.get_predictions()
