#!/usr/bin/env python
import itertools
import json
import numpy
import os
import unittest
from lib.abstract_recommender import AbstractRecommender
from lib.content_based import ContentBased
from lib.collaborative_filtering import CollaborativeFiltering
from lib.evaluator import Evaluator
from lib.recommender_system import RecommenderSystem
from util.abstracts_preprocessor import AbstractsPreprocessor
from util.data_parser import DataParser
from util.model_initializer import ModelInitializer


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        self.documents, self.users = 8, 10
        documents_cnt, users_cnt = self.documents, self.users
        self.n_iterations = 5
        self.n_factors = 5
        self.hyperparameters = {'n_factors': self.n_factors}
        self.options = {'n_iterations': self.n_iterations}
        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.n_iterations)

        def mock_process(self=None):
            pass

        def mock_get_abstracts(self=None):
            return {0: 'hell world berlin dna evolution', 1: 'freiburg is green',
                    2: 'the best dna is the dna of dinasours', 3: 'truth is absolute',
                    4: 'berlin is not that green', 5: 'truth manifests itself',
                    6: 'plato said truth is beautiful', 7: 'freiburg has dna'}

        def mock_get_ratings_matrix(self=None):
            return [[int(not bool((article + user) % 3)) for article in range(documents_cnt)]
                    for user in range(users_cnt)]

        def mock_get_word_distribution(self=None):
            abstracts = mock_get_abstracts()
            vocab = set(itertools.chain(*list(map(lambda ab: ab.split(' '), abstracts.values()))))
            w2i = dict(zip(vocab, range(len(vocab))))
            word_to_count = [(w2i[word], sum(abstract.split(' ').count(word)
                                             for doc_id, abstract in abstracts.items())) for word in vocab]
            article_to_word = list(set([(doc_id, w2i[word])
                                        for doc_id, abstract in abstracts.items() for word in abstract.split(' ')]))
            article_to_word_to_count = list(set([(doc_id, w2i[word], abstract.count(word))
                                                 for doc_id, abstract in abstracts.items()
                                                 for word in abstract.split(' ')]))
            return word_to_count, article_to_word, article_to_word_to_count

        abstracts = mock_get_abstracts()
        word_to_count, article_to_word,  article_to_word_to_count = mock_get_word_distribution()
        self.abstracts_preprocessor = AbstractsPreprocessor(abstracts, word_to_count,
                                                            article_to_word, article_to_word_to_count)
        self.ratings_matrix = numpy.array(mock_get_ratings_matrix())
        self.evaluator = Evaluator(self.ratings_matrix, self.abstracts_preprocessor)
        setattr(DataParser, "get_abstracts", mock_get_abstracts)
        setattr(DataParser, "process", mock_process)
        setattr(DataParser, "get_ratings_matrix", mock_get_ratings_matrix)
        setattr(DataParser, "get_word_distribution", mock_get_word_distribution)


class TestRecommenderSystem(TestcaseBase):
    def runTest(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(base_dir), 'config/recommender.json')) as data_file:
            json_config = json.load(data_file)
        rec_system = RecommenderSystem()
        self.assertEqual(rec_system.hyperparameters, json_config['recommender']['hyperparameters'])
        self.assertEqual(rec_system.config.config_dict, json_config['recommender'])
        n_factors = self.n_factors
        rec_system.initializer.config['n_factors'] = n_factors
        rec_system.content_based.n_factors = n_factors
        rec_system.content_based.hyperparameters['n_factors'] = n_factors
        rec_system.collaborative_filtering.n_factors = n_factors
        rec_system.collaborative_filtering.hyperparameters['n_factors'] = n_factors
        self.assertTrue(isinstance(rec_system.evaluator, Evaluator))
        self.assertTrue(isinstance(rec_system.content_based, ContentBased))
        self.assertTrue(isinstance(rec_system.collaborative_filtering, CollaborativeFiltering))
        self.assertTrue(isinstance(rec_system.content_based, AbstractRecommender))
        if rec_system.config.config_dict['recommender'] == 'userbased':
            self.assertTrue(isinstance(rec_system.recommender, CollaborativeFiltering))
        if rec_system.config.config_dict['recommender'] == 'itembased':
            self.assertTrue(isinstance(rec_system.recommender, ContentBased))
        self.assertEqual(rec_system.content_based.n_items, self.documents)
        self.assertEqual(rec_system.content_based.n_factors, n_factors)
        rec_system.train()
        self.assertLessEqual(rec_system.content_based.get_document_topic_distribution().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(rec_system.content_based.get_document_topic_distribution().min(), -1e-6)
        self.assertEqual(rec_system.content_based.get_document_topic_distribution().shape, (self.documents, n_factors))
