#!/usr/bin/env python
import json
import numpy
import os
import unittest
from lib.content_based import ContentBased
from lib.recommender_system import RecommenderSystem
from util.recommender_configuer import RecommenderConfiguration


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        pass


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


class TestContentBased(TestcaseBase):
    def runTest(self):
        content_based = ContentBased(numpy.zeros((4, 20)), 5)
        self.assertEqual(content_based.n_factors, 5)
        self.assertEqual(content_based.n_items, 4)
        self.assertEqual(content_based.ratings.shape, (4, 20))
        content_based.train()
        self.assertEqual(content_based.get_word_distribution().shape, (4, 5))


class TestRecommenderSystem(TestcaseBase):
    def runTest(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(base_dir), 'config/recommender.json')) as data_file:
            json_config = json.load(data_file)
        rec_system = RecommenderSystem(10)
        self.assertEqual(rec_system.hyperparameters, json_config['recommender']['hyperparameters'])
        self.assertEqual(rec_system.config.config_dict, json_config['recommender'])
        self.assertTrue(isinstance(rec_system.content_based, ContentBased))
        # self.assertTrue(isinstance(rec_system.collaborative_filtering, CollaborativeFiltering))
