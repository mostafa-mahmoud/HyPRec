#!/usr/env/bin python
"""
This module is used to load configurations for recommenders.
"""
import json
import os


class RecommenderConfiguration(object):
    """
    A class that will be used to setup the configuration of a RecommenderSystem.
    """
    def __init__(self, config=None):
        """
        Constructs a configuration from the config/ directory.
        """
        if not config:
            base_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(os.path.dirname(base_dir), 'config/recommender.json')) as data_file:
                self.config_dict = json.load(data_file)['recommender']
        else:
            self.config_dict = config

    def get_content_based(self):
        """
        Get the configuration of the content-based algorithm.

        :returns: A string of the algorithm's name.
        :rtype: str
        """
        return self.config_dict['content-based']

    def get_collaborative_filtering(self):
        """
        Get the configuration of the collaborative-filtering algorithm.

        :returns: A string of the algorithm's name.
        :rtype: str
        """
        return self.config_dict['collaborative-filtering']

    def get_options(self):
        """
        Get the additional options of the recommender.

        :returns: A dictionary of the options.
        :rtype: dict
        """
        if 'options' in self.config_dict.keys():
            return self.config_dict['options']
        else:
            return {}

    def get_error_metric(self):
        """
        Get the configuration of the error metric.

        :returns: A string of the metric's name.
        :rtype: str
        """
        return self.config_dict['error-metric']

    def get_hyperparameters(self):
        """
        Get the hyperparameters.

        :returns: A dictionary of the hyperparameters.
        :rtype: dict
        """
        return self.config_dict['hyperparameters']
