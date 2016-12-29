#!/usr/bin/env python
import numpy
from lib.content_based import ContentBased
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDARecommender(ContentBased):
    def __init__(self, abstracts, n_factors, n_iterations=5):
        self.n_factors = n_factors
        self.n_items = len(abstracts)
        self.abstracts = abstracts
        self.n_iterations = n_iterations

    def train(self):
        term_freq_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=self.n_items)
        term_freq = term_freq_vectorizer.fit_transform(self.abstracts)

        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=self.n_iterations,
                                        learning_method='online', learning_offset=50., random_state=0)
        self.word_distribution = lda.fit_transform(term_freq)
        # tf_feature_names = tf_vectorizer.get_feature_names()
        # print([['%.3f:%s' % (lda.components_[topic_idx][i], tf_feature_names[i]) for i in topic.argsort()]
        #                                        for topic_idx, topic in enumerate(dist)])

    def get_word_distribution(self):
        return self.word_distribution
