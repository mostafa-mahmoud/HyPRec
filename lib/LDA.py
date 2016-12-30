#!/usr/bin/env python
import numpy
from lib.content_based import ContentBased
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDARecommender(ContentBased):
    def __init__(self, abstracts, evaluator, config, verbose=False):
        super(LDARecommender, self).__init__(abstracts, evaluator, config, verbose)

    def train(self, n_iter=5):
        term_freq_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=self.n_items)
        term_freq = term_freq_vectorizer.fit_transform(self.abstracts)

        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=n_iter,
                                        learning_method='online', learning_offset=50., random_state=0)
        self.word_distribution = lda.fit_transform(term_freq)

    def split(self):
        super(LDARecommender, self).split()

    def set_config(self, config):
        """
        set the hyperparamenters of the algorithm.
        """
        super(LDARecommender, self).set_config(config)

    def get_word_distribution(self):
        """
        @returns a matrix of the words x topics distribution.
        """
        return self.word_distribution
