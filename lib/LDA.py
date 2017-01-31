#!/usr/bin/env python
"""
A module that contains the content-based recommender LDARecommender that uses
LDA.
"""
import itertools
from lib.content_based import ContentBased
from sklearn.decomposition import LatentDirichletAllocation


class LDARecommender(ContentBased):
    """
    LDA Recommender, a content based recommender that uses LDA.
    """
    def __init__(self, initializer, abstracts_preprocessor, ratings, evaluator, config,
                 verbose=False, load_matrices=True, dump=True):
        """
        Constructor of ContentBased processor.

        :param ModelInitializer initializer: A model initializer.
        :param AbstractsProprocessor abstracts_preprocessor: Abstracts preprocessor.
        :param Ratings ratings: Ratings matrix
        :param Evaluator evaluator: An evaluator object.
        :param dict config: A dictionary of the hyperparameters.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump: A flag for saving the matrices.
        """
        super(LDARecommender, self).__init__(initializer, abstracts_preprocessor, ratings, evaluator, config,
                                             verbose, load_matrices, dump)

    def train(self, n_iter=5):
        """
        Try to load saved matrix if load_matrices is false, else train

        :param int n_iter: The number of iterations of the training the model.
        """
        # Try to read from file.
        matrix_found = False
        if self.load_matrices is True:
            matrix_shape = (self.abstracts_preprocessor.get_num_items(), self.config['n_factors'])
            matrix_found, matrix = self.initializer.load_matrix(self.config, 'document_distribution_lda', matrix_shape)
            self.document_distribution = matrix
            if self._v and matrix_found:
                print("Document distribution was set from file, will not train.")
        if matrix_found is False:
            if self._v and self.load_matrices:
                print("Document distribution file was not found, will train LDA.")
            self._train(n_iter)

    def _train(self, n_iter):
        """
        Train LDA Recommender, and store the document_distribution.

        :param int n_iter: The number of iterations of training the model.
        """
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix()
        if self._v:
            print('...')
            print(term_freq.todense())
            for tup in itertools.chain(*[
                    list(zip(map(lambda t: (doc_id, t), row.indices), row.data))
                    for doc_id, row in enumerate(term_freq)]):
                print(tup)
            print('...')

        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=n_iter,
                                        learning_method='online', learning_offset=50., random_state=0)
        if self._v:
            print("Initialized LDA model..., Training LDA...")

        self.document_distribution = lda.fit_transform(term_freq)
        if self.dump:
            self.initializer.save_matrix(self.document_distribution, 'document_distribution_lda')
        if self._v:
            print("LDA trained..")

    def naive_split(self):
        """
        split the data into train and test data.
        :returns: A tuple of (train_data, test_data)
        :rtype: tuple
        """
        return super(LDARecommender, self).split()

    def set_config(self, config):
        """
        set the hyperparamenters of the algorithm.

        :param dict config: A dictionary of the hyperparameters.
        """
        super(LDARecommender, self).set_config(config)

    def get_document_topic_distribution(self):
        """
        Get the matrix of document X topics distribution.

        :returns: A matrix of documents X topics distribution.
        :rtype: ndarray
        """
        return self.document_distribution
