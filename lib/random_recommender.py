"""
Module for random Recommender
"""
import numpy
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender


class RandomRecommender(AbstractRecommender):
    """
    A class that takes in the rating matrix and oupits random predictions
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True, train_more=True,
                 is_hybrid=False, update_with_items=False, init_with_content=True):
        """
        Constructor of the random recommender.

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: Evaluator of the recommender and holder of the input data.
        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        :param dict options: Dictionary of the run options, contains n_iterations and k_folds
        :param boolean verbose: A flag if True, tracing will be printed
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        :param boolean is_hybrid: A flag indicating whether the recommender is hybrid or not.
        :param boolean update_with_items: A flag the decides if we will use the items matrix in the update rule.
        """
        # setting input
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.get_ratings()
        self.n_users, self.n_items = self.ratings.shape
        self.k_folds = None
        self.prediction_fold = -1

        # setting flags
        self._verbose = verbose
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._train_more = train_more
        self._is_hybrid = is_hybrid
        self._update_with_items = update_with_items
        self._split_type = 'user'
        self._init_with_content = init_with_content

        self.set_hyperparameters(hyperparameters)
        self.set_options(options)

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        Deprecated(code from collaborative filtering)
        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.predictions = None
        self.hyperparameters = hyperparameters.copy()

    @overrides
    def train(self):
        """
        Setting the data and printing the evaluation report
        """
        if self.splitting_method == 'naive':
            self.set_data(*self.evaluator.naive_split(self._split_type))
        else:
            self.set_data(*self.evaluator.naive_split(self._split_type))
            self.fold_test_indices = self.evaluator.get_kfold_indices()
        return self.get_evaluation_report()

    @overrides
    def get_predictions(self):
        """
        Predict random ratings for every user and item.
        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        if self.predictions is None:
            self.predictions = numpy.random.random_sample((self.n_users, self.n_items))
        return self.predictions
