#!/usr/bin/env python
"""
A module that contains the content-based recommender LDARecommender that uses
LDA.
"""
from lib.content_based import ContentBased
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDARecommender(ContentBased):
    """
    LDA Recommender, a content based recommender that uses LDA.
    """
    def __init__(self, abstracts, evaluator, config, verbose=False):
        """
        Constructor of ContentBased processor.
        @param (list[str]) abstracts: List of the texts of the abstracts of the papers.
        @param (Evaluator) evaluator: An evaluator object.
        @param (dict) config: A dictionary of the hyperparameters.
        @param (boolean) verbose: A flag for printing while computing.
        """
        super(LDARecommender, self).__init__(abstracts, evaluator, config, verbose)

    def train(self, n_iter=5):
        """
        Train LDA Recommender, and store the document_distribution.
        """
        term_freq_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english',
                                               max_features=self.n_items)
        term_freq = term_freq_vectorizer.fit_transform(self.abstracts)

        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=n_iter,
                                        learning_method='online', learning_offset=50., random_state=0)
        self.document_distribution = lda.fit_transform(term_freq)

    def split(self):
        """
        split the data into train and test data.
        @returns (tuple) A tuple of (train_data, test_data)
        """
        return super(LDARecommender, self).split()

    def set_config(self, config):
        """
        set the hyperparamenters of the algorithm.
        @param (dict) config: A dictionary of the hyperparameters.
        """
        super(LDARecommender, self).set_config(config)

    def get_document_topic_distribution(self):
        """
        @returns (ndarray) A matrix of documents X topics distribution.
        """
        return self.document_distribution
