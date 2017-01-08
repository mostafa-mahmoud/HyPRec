#!/usr/bin/env python


class AbstractRecommender(object):
    """
    A class that acts like an interface, it is never initialized but the uv_decomposition
    and content_based should implement it's methods.
    """
    def __init__(self):
        raise NotImplementedError("Can't initialize this class")

    def train(self):
        raise NotImplementedError("Can't call this method")

    def split(self):
        raise NotImplementedError("Can't call this method")

    def set_config(self):
        raise NotImplementedError("Can't call this method")
