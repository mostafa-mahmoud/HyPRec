#!/usr/bin/env python
"""
This module provides main functionalities to store users' top recommendations.
"""
import bisect


class TopRecommendations(object):
    """
    A class that will store top recommendations for user and provide functionalities for
    inserting the recommendations.
    """
    def __init__(self, n_recommendations):
        """
        Constructor of top recommendations.
        @param (int) n_recommendations: number of the recommendations to be stored.
        """
        self.recommendations_values = []
        self.recommendations_indices = []
        self.n_recommendations = n_recommendations

    def insert(self, index, value):
        """
        The function inserts the recommendation value and index in 2 parallel arrays. It inserts while keeping
        the values array sorted and without exceeding the n_recommendations size.
        @param (int) index: index of the recommended item in the ratings matrix.
        @param (float) value: predicted recommendation score.
        """
        if self.n_recommendations > len(self.recommendations_values):
            inserted_index = self._insert_and_return_index(self.recommendations_values, value)
            self.recommendations_indices = self._insert_at_index(self.recommendations_indices, index, inserted_index)
        elif len(self.recommendations_values) != 0:
            if self.recommendations_values[0] < value:
                self.recommendations_values.pop(0)
                self.recommendations_indices.pop(0)
                inserted_index = self._insert_and_return_index(self.recommendations_values, value)
                self.recommendations_indices = self._insert_at_index(self.recommendations_indices,
                                                                     index, inserted_index)

    def _insert_and_return_index(self, arr, val):
        """
        The method is only used internally, it inserts a value to an array while keeping it sorted and returns
        the index where the value was inserted.
        @param (list) arr: list of floats representing the recommendation score.
        @param (float) val: a float representing the recommendation value to be inserted.
        @return (int) the index of the inserted value.
        """
        index = bisect.bisect(arr, val)
        bisect.insort(arr, val)
        return index

    def _insert_at_index(self, arr, val, index):
        """
        The method is only used internally, it inserts in item in an array at a given index and shifts the
        values after that index to the right.
        @param (list) arr: list of floats representing the recommendation score.
        @param (float) val: a float representing the recommendation value to be inserted.
        @param (int) index: index where the value will be inserted.
        @returns (list) an array with the inserted value.
        """
        return arr[0: index] + [val] + arr[index:]

    def get_indices(self):
        """
        getter for the indices array.
        @return (list) list of indices of the recommendations.
        """
        return self.recommendations_indices

    def get_values(self):
        """
        getter for the values array.
        @returns (list) list of the values of the recommendations.
        """
        return self.recommendations_values

    def get_recommendations_count(self):
        """
        getter for the recommendations count.
        @returns (int) integer representing number of the recommendations currently stored.
        """
        return len(self.recommendations_values)
