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
from util.model_initializer import ModelInitializer
from util.runs_loader import RunsLoader


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
        self.assertEqual(config.get_recommender(), json_config['recommender']['recommender'])
        self.assertEqual(config.get_description(), json_config['recommender']['desc'])
        config = RecommenderConfiguration(json_config)
        self.assertEqual(config.get_content_based(), json_config['recommender']['content-based'])
        self.assertEqual(config.get_collaborative_filtering(), json_config['recommender']['collaborative-filtering'])
        self.assertEqual(config.get_error_metric(), json_config['recommender']['error-metric'])
        self.assertEqual(config.get_options(), json_config['recommender']['options'])
        self.assertEqual(config.get_hyperparameters(), json_config['recommender']['hyperparameters'])
        self.assertEqual(config.get_recommender(), json_config['recommender']['recommender'])
        self.assertEqual(config.get_description(), json_config['recommender']['desc'])


class TestRunsLoader(TestcaseBase):
    def runTest(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(base_dir), 'config/runs.json')) as data_file:
            json_config = json.load(data_file)
        runs_loader = RunsLoader()
        self.assertEqual(runs_loader.get_runnable_recommenders(), json_config['runs'])
        for run in runs_loader.get_runnable_recommenders():
            self.assertTrue('recommender' in run.keys())
            self.assertEqual(set(run['recommender'].keys()), set(['content-based', 'collaborative-filtering',
                                                                  'error-metric', 'recommender',
                                                                  'hyperparameters', 'options', 'desc']))


class TestAbstractsPreprocessor(TestcaseBase):
    def runTest(self):
        self.assertEqual(len(set(itertools.chain(*list(map(lambda ab: ab.split(' '), self.abstracts.values()))))),
                         self.abstracts_preprocessor.get_num_vocab())
        self.assertEqual(set(self.abstracts.values()), set(self.abstracts_preprocessor.get_abstracts()))
        self.assertEqual(self.word_to_count, self.abstracts_preprocessor.get_word_to_counts())
        self.assertEqual(self.article_to_word, self.abstracts_preprocessor.get_article_to_words())
        self.assertEqual(self.article_to_word_to_count, self.abstracts_preprocessor.get_article_to_word_to_count())
        self.assertEqual(self.documents, self.abstracts_preprocessor.get_num_items())
        self.assertTrue(isinstance(self.abstracts_preprocessor.get_term_frequency_sparse_matrix(), sparse.csr_matrix))
        self.assertEqual(max(map(lambda inp: len(inp.split(' ')), self.abstracts_preprocessor.abstracts.values())),
                         self.abstracts_preprocessor.get_num_units())


class TestModelInitializer(TestcaseBase):
    def runTest(self):
        users_cnt, documents_cnt = self.users, self.documents
        config = RecommenderConfiguration().get_hyperparameters()
        config['n_factors'] = 5
        initializer = ModelInitializer(config, 1)
        path = initializer._create_path('user_v', (users_cnt, documents_cnt))
        self.assertTrue(path.endswith('n_iterations-1,n_rows-10user_v.dat'))
        matrix_shape = (users_cnt, config['n_factors'])
        users_mat = numpy.random.random(matrix_shape)
        initializer.save_matrix(users_mat, 'user_v')
        self.assertTrue(os.path.isfile(path))
        loaded, loaded_matrix = initializer.load_matrix(config, 'user_v', matrix_shape)
        self.assertTrue(loaded)
        self.assertTrue(numpy.alltrue(loaded_matrix == users_mat))
