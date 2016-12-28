#!/usr/bin/env python
import numpy
from content_based import ContentBased
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDARecommender(ContentBased):
    def __init__(self, abstracts, n_factors, n_iterations=5):
        self.n_factors = n_factors
        self.n_items = len(abstracts)
        self.abstracts = abstracts
        self.n_iterations = n_iterations

    def train(self):
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')  # max_features=n_features,
        tf = tf_vectorizer.fit_transform(self.abstracts)

        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=self.n_iterations,
                                        learning_method='online', learning_offset=50., random_state=0)
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        self.word_distribution = numpy.matrix([[lda.components_[topic_idx][i] for i in topic.argsort()]
                                               for topic_idx, topic in enumerate(lda.components_)]).T

    def get_word_distribution(self):
        return self.word_distribution
