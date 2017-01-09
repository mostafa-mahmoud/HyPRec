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
    def __init__(self, abstracts_preprocessor, evaluator, config, verbose=False):
        """
        Constructor of ContentBased processor.

        :param AbstractsProprocessor abstracts_preprocessor: Abstracts preprocessor
        :param Evaluator evaluator: An evaluator object.
        :param dict config: A dictionary of the hyperparameters.
        :param boolean verbose: A flag for printing while computing.
        """
        super(LDARecommender, self).__init__(abstracts_preprocessor, evaluator, config, verbose)

    def train(self, n_iter=5):
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
        if self._v:
            print("LDA trained..")

    def split(self):
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
