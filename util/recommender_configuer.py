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
            self.config_dict = config['recommender']

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

    def get_recommender(self):
        """
        Get the recommender type

        :returns: A string of userbased or itembased recommendations.
        :rtype: str
        """
        return self.config_dict['recommender']

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

    def get_all_config(self):
        """
        Get all configs

        :returns: A dictionary of the whole json configuration
        :rtype: dict
        """
        return {'recommender': self.config_dict}

    def set_recommender_type(self, recommender_type='itembased'):
        """
        Set the hyperparameters.

        :param str recommender: A string 'itembased' or 'userbased'
        """
        self.config_dict['recommender'] = recommender_type

    def get_hyperparameters(self):
        """
        Get the hyperparameters.

        :returns: A dictionary of the hyperparameters.
        :rtype: dict
        """
        return self.config_dict['hyperparameters']
