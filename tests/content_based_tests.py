#!/usr/bin/env python
import itertools
import numpy
import unittest
from lib.abstract_recommender import AbstractRecommender
from lib.content_based import ContentBased
from lib.evaluator import Evaluator
from lib.LDA import LDARecommender
from lib.LDA2Vec import LDA2VecRecommender
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


class TestContentBased(TestcaseBase):
    def runTest(self):
        content_based = ContentBased(self.initializer, self.evaluator, self.hyperparameters, self.options)
        self.assertEqual(content_based.n_factors, self.n_factors)
        self.assertEqual(content_based.n_items, self.documents)
        content_based.train()
        self.assertEqual(content_based.get_document_topic_distribution().shape, (self.documents, self.n_factors))
        self.assertTrue(isinstance(content_based, AbstractRecommender))
        self.assertTrue(content_based.get_predictions().shape, (self.users, self.documents))
        self.assertLessEqual(content_based.get_predictions().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(content_based.get_predictions().min(), -1e-6)


class TestLDA(TestcaseBase):
    def runTest(self):
        content_based = LDARecommender(self.initializer, self.evaluator, self.hyperparameters, self.options)
        self.assertEqual(content_based.n_factors, self.n_factors)
        self.assertEqual(content_based.n_items, self.documents)
        content_based.train()
        self.assertEqual(content_based.get_document_topic_distribution().shape, (self.documents, self.n_factors))
        self.assertLessEqual(content_based.get_document_topic_distribution().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(content_based.get_document_topic_distribution().min(), -1e-6)
        self.assertTrue(isinstance(content_based, AbstractRecommender))
        self.assertEqual(content_based.get_predictions().shape, (self.users, self.documents))
        self.assertLessEqual(content_based.get_predictions().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(content_based.get_predictions().min(), -1e-6)


class TestLDA2Vec(TestcaseBase):
    def runTest(self):
        content_based = LDA2VecRecommender(self.initializer, self.evaluator, self.hyperparameters, self.options)
        self.assertEqual(content_based.n_factors, self.n_factors)
        self.assertEqual(content_based.n_items, self.documents)
        content_based.train()
        self.assertEqual(content_based.get_document_topic_distribution().shape, (self.documents, self.n_factors))
        self.assertLessEqual(content_based.get_document_topic_distribution().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(content_based.get_document_topic_distribution().min(), -1e-6)
        self.assertTrue(isinstance(content_based, AbstractRecommender))
        self.assertEqual(content_based.get_predictions().shape, (self.users, self.documents))
        self.assertLessEqual(content_based.get_predictions().max(), 1.0 + 1e-6)
        self.assertGreaterEqual(content_based.get_predictions().min(), -1e-6)
