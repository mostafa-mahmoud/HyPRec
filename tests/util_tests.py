#!/usr/bin/env python
import itertools
import json
import numpy
import os
import unittest
from scipy import sparse
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration
from util.abstracts_preprocessor import AbstractsPreprocessor


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 8, 10
        documents_cnt, users_cnt = self.documents, self.users

        def mock_process(self=None):
            pass

        def mock_get_abstracts(self=None):
            return {1: 'hell world berlin dna evolution', 2: 'freiburg is green',
                    3: 'the best dna is the dna of dinasours', 4: 'truth is absolute',
                    5: 'berlin is not that green', 6: 'truth manifests itself',
                    7: 'plato said truth is beautiful', 8: 'freiburg has dna'}

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]

        def mock_get_word_distribution(self=None):
            abstracts = mock_get_abstracts()
            vocab = set(itertools.chain(*list(map(lambda ab: ab.split(' '), abstracts.values()))))
            w2i = dict(zip(vocab, range(1, len(vocab) + 1)))
            word_to_count = [(w2i[word], sum(abstract.split(' ').count(word)
                                             for doc_id, abstract in abstracts.items())) for word in vocab]
            article_to_word = list(set([(doc_id, w2i[word])
                                        for doc_id, abstract in abstracts.items() for word in abstract.split(' ')]))
            article_to_word_to_count = list(set([(doc_id, w2i[word], abstract.count(word))
                                                 for doc_id, abstract in abstracts.items()
                                                 for word in abstract.split(' ')]))
            return word_to_count, article_to_word, article_to_word_to_count

        self.abstracts = mock_get_abstracts()
        self.word_to_count, self.article_to_word, self.article_to_word_to_count = mock_get_word_distribution()
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        self.abstracts_preprocessor = AbstractsPreprocessor(mock_get_abstracts(), *mock_get_word_distribution())
        setattr(DataParser, "get_abstracts", mock_get_abstracts)
        setattr(DataParser, "process", mock_process)
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)
        setattr(DataParser, "get_word_distribution", mock_get_ratings_matrix)


class TestRecommenderConfiguration(TestcaseBase):
    def runTest(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(base_dir), 'config/recommender.json')) as data_file:
            json_config = json.load(data_file)
        config = RecommenderConfiguration()
        self.assertEqual(config.get_content_based(), json_config['recommender']['content-based'])
        self.assertEqual(config.get_collaborative_filtering(), json_config['recommender']['collaborative-filtering'])
        self.assertEqual(config.get_error_metric(), json_config['recommender']['error-metric'])
        self.assertEqual(config.get_options(), json_config['recommender']['options'])
        self.assertEqual(config.get_hyperparameters(), json_config['recommender']['hyperparameters'])


class TestAbstractsPreprocessor(TestcaseBase):
    def runTest(self):
        self.assertEqual(len(set(itertools.chain(*list(map(lambda ab: ab.split(' '), self.abstracts.values()))))),
                         self.abstracts_preprocessor.get_num_vocab())
        self.assertEqual(set(self.abstracts.values()), set(self.abstracts_preprocessor.get_abstracts()))
        self.assertEqual(list(map(lambda t: (t[0] - 1, t[1]), self.word_to_count)),
                         self.abstracts_preprocessor.get_word_to_counts())
        self.assertEqual(list(map(lambda t: (t[0] - 1, t[1] - 1), self.article_to_word)),
                         self.abstracts_preprocessor.get_article_to_words())
        self.assertEqual(list(map(lambda t: (t[0] - 1, t[1] - 1, t[2]), self.article_to_word_to_count)),
                         self.abstracts_preprocessor.get_article_to_word_to_count())
        self.assertEqual(self.documents, self.abstracts_preprocessor.get_num_items())
        self.assertTrue(isinstance(self.abstracts_preprocessor.get_term_frequency_sparse_matrix(), sparse.csr_matrix))
        self.assertEqual(max(map(lambda inp: len(inp.split(' ')), self.abstracts_preprocessor.abstracts.values())),
                         self.abstracts_preprocessor.get_num_units())
