#!/usr/bin/env python
"""
A module that contains a User object that will be used in recommendations
and data parsing.
"""


class User(object):
    def __init__(self, user_id):
        """
        Constructor for user.
        """
        self.user_id = user_id
