#!/usr/env/bin python
"""
This module is used to initalize a set of runnable recommenders according to different configurations.
"""
import json
import os


class RunsLoader(object):
    """
    """
    def __init__(self, runs=None):
        """
        Initalize the RunsLoader by the runs dictionary.
        """
        if runs is None:
            base_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(os.path.dirname(base_dir), 'config/runs.json')) as data_file:
                self.runs = json.load(data_file)['runs']
        else:
            self.runs = runs['runs']

    def get_runnable_recommenders(self):
        """
        Get the list of recommender configurations.

        :returns: A list of recommender configurations.
        :rtype: List[dict]
        """
        return self.runs
