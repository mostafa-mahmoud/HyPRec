#!/usr/bin/env python
"""
This is a module that contains the main class and functionalities of the recommender systems.
"""
import numpy
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender
from lib.collaborative_filtering import CollaborativeFiltering
from lib.content_based import ContentBased
from lib.evaluator import Evaluator
from lib.LDA import LDARecommender
from lib.LDA2Vec import LDA2VecRecommender
from lib.SDAE import SDAERecommender
from util.abstracts_preprocessor import AbstractsPreprocessor
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration
from util.model_initializer import ModelInitializer


class RecommenderSystem(AbstractRecommender):
    """
    A class that will combine the content-based and collaborative-filtering,
    in order to provide the main functionalities of recommendations.
    """
    def __init__(self, initializer=None, abstracts_preprocessor=None, ratings=None, config=None,
                 process_parser=False, verbose=False, load_matrices=True, dump_matrices=True, train_more=True,
                 random_seed=False, results_file_name='top_recommendations'):
        """
        Constructor of the RecommenderSystem.

        :param ModelInitializer initializer: A model initializer.
        :param AbstractsPreprocessor abstracts_preprocessor: A preprocessor of abstracts, if None then queried.
        :param int[][] ratings: Ratings matrix; if None, matrix gets queried from the database.
        :param boolean process_parser: A Flag deceiding process the dataparser.
        :param boolean verbose: A flag deceiding to print progress.
        :param boolean dump_matrices: A flag for saving output matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        :param boolean random_seed: A flag to determine if we will use random seed or not.
        :param str results_file_name: Top recommendations results' file name
        """
        if process_parser:
            DataParser.process()

        if ratings is None:
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
        else:
            self.ratings = ratings

        if abstracts_preprocessor is None:
            self.abstracts_preprocessor = AbstractsPreprocessor(DataParser.get_abstracts(),
                                                                *DataParser.get_word_distribution())
        else:
            self.abstracts_preprocessor = abstracts_preprocessor

        # Get configurations
        self.config = RecommenderConfiguration(config)

        # Set flags
        self.results_file_name = results_file_name + '.dat'
        self._verbose = verbose
        self._dump_matrices = dump_matrices
        self._load_matrices = load_matrices
        self._train_more = train_more
        self._split_type = 'user'
        self._random_seed = random_seed

        self.set_hyperparameters(self.config.get_hyperparameters())
        self.set_options(self.config.get_options())

        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iter, self._verbose)

        if self.config.get_error_metric() == 'RMS':
            self.evaluator = Evaluator(self.ratings, self.abstracts_preprocessor, self._random_seed, self._verbose)
        else:
            raise NameError("Not a valid error metric %s. Only option is 'RMS'" % self.config.get_error_metric())

        # Initialize content based.
        if self.config.get_content_based() == 'None':
            self.content_based = ContentBased(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                              self._verbose, self._load_matrices, self._dump_matrices)
        elif self.config.get_content_based() == 'LDA':
            self.content_based = LDARecommender(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                                self._verbose, self._load_matrices, self._dump_matrices)
        elif self.config.get_content_based() == 'LDA2Vec':
            self.content_based = LDA2VecRecommender(self.initializer, self.evaluator, self.hyperparameters,
                                                    self.options, self._verbose,
                                                    self._load_matrices, self._dump_matrices)
        else:
            raise NameError("Not a valid content based %s. Options are 'None', "
                            "'LDA', 'LDA2Vec'" % self.config.get_content_based())

        # Initialize collaborative filtering.
        if self.config.get_collaborative_filtering() == 'ALS':
            is_hybrid = self.config.get_recommender() == 'hybrid'
            if self.config.get_content_based() == 'None':
                raise NameError("Not valid content based 'None' with hybrid recommender")
            self.collaborative_filtering = CollaborativeFiltering(self.initializer, self.evaluator,
                                                                  self.hyperparameters, self.options,
                                                                  self._verbose, self._load_matrices,
                                                                  self._dump_matrices, self._train_more,
                                                                  is_hybrid)
        elif self.config.get_collaborative_filtering() == 'SDAE':
            self.collaborative_filtering = SDAERecommender(self.initializer, self.evaluator, self.hyperparameters,
                                                           self.options, self._verbose, self._load_matrices,
                                                           self._dump_matrices)
            if not self.config.get_content_based() == 'None':
                raise NameError("Not a valid content based %s with SDAE. You can only use 'None'"
                                % self.config.get_content_based())
        elif self.config.get_collaborative_filtering() == 'None':
            if not self.config.get_recommender() == 'itembased':
                raise NameError("None collaborative filtering is only valid with itembased recommender type")
            elif self.config.get_content_based() == 'None':
                raise NameError("Not valid content based 'None' with item-based recommender")
            self.collaborative_filtering = None
        else:
            raise NameError("Not a valid collaborative filtering %s. "
                            "Only options are 'None', 'ALS', 'SDAE'" % self.config.get_collaborative_filtering())

        # Initialize recommender
        if self.config.get_recommender() == 'itembased':
            self.recommender = self.content_based
        elif self.config.get_recommender() == 'userbased':
            self.recommender = self.collaborative_filtering
        elif self.config.get_recommender() == 'hybrid':
            self.recommender = self
        else:
            raise NameError("Invalid recommender type %s. "
                            "Only options are 'userbased','itembased', and 'hybrid'" % self.config.get_recommender())

    @overrides
    def set_options(self, options):
        """
        Set the options of the recommender. Namely n_iterations and k_folds.

        :param dict options: A dictionary of the options.
        """
        self.n_iter = options['n_iterations']
        self.options = options.copy()

    @overrides
    def get_evaluation_report(self):
        if self.config.get_recommender() == 'hybrid':
            return self.collaborative_filtering.get_evaluation_report()
        return self.recommender.get_evaluation_report()

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        Setter of the hyperparameters of the recommender.

        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.predictions = None
        self.hyperparameters = hyperparameters.copy()
        if hasattr(self, 'collaborative_filtering') and self.collaborative_filtering is not None:
            self.collaborative_filtering.set_hyperparameters(hyperparameters)
        if hasattr(self, 'content_based') and self.content_based is not None:
            self.content_based.set_hyperparameters(hyperparameters)

    @overrides
    def train(self):
        """
        Train the recommender on the given data.

        :returns: The error of the predictions.
        :rtype: float
        """
        assert(self.recommender == self.collaborative_filtering or
               self.recommender == self.content_based or self.recommender == self)
        if self._verbose:
            print("Training content-based %s..." % self.content_based)
        content_based_error = numpy.inf
        if self.content_based.__class__ != ContentBased:
            self.content_based.train()
            self.content_based.get_predictions()
            # Optimize unused memory
            if not self.recommender == self.content_based:
                del self.content_based.train_data
                del self.content_based.test_data
            if hasattr(self.content_based, 'fold_test_indices'):
                del self.content_based.fold_test_indices
        if self.recommender == self.collaborative_filtering:
            theta = None
            if self.content_based.get_document_topic_distribution() is not None:
                theta = self.content_based.get_document_topic_distribution().copy()
            if self._verbose:
                print("Training collaborative-filtering %s..." % self.collaborative_filtering)
            if theta is None:
                return self.collaborative_filtering.train()
            else:
                return self.collaborative_filtering.train(theta)
        elif self.recommender == self:
            if self._verbose:
                print("Training collaborative_filtering %s..." % self.collaborative_filtering)
            self.collaborative_filtering.set_item_based_recommender(self.content_based)
            return self.collaborative_filtering.train()
        self.predictions = self.recommender.get_predictions()
        return content_based_error

    @overrides
    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        if self.predictions is None:
            if self.recommender == self:
                return self.collaborative_filtering.get_predictions()
            else:
                self.predictions = self.recommender.get_predictions()
        return self.predictions
