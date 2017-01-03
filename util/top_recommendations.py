import bisect

class TopRecommendations(object):

    def __init__(self, n_recommendations):
        self.recommendations_values = []
        self.recommendations_indices = []
        self.n_recommendations = n_recommendations

    def insert(self, index, value):
        if self.n_recommendations > len(self.recommendations_values)
            inserted_index = self._insert_and_return_index(self.recommendations_values, value)
            self._insert_at_index(self.recommendations_indices, index, inserted_index)
        if len(self.recommendations_values != 0):
            if self.recommendations_values[0] < value:
                inserted_index = self._insert_and_return_index(self.recommendations_values, value)
                self._insert_at_index(self.recommendations_indices, index, inserted_index)

    def _insert_and_return_index(self, arr, val):
        index = bisect.bisect(arr, val)
        bisect.insort(val)
        return index

    def _insert_at_index(self, arr, val, index):
        return arr[0: index] + [val] + arr[index: ]

    def get_indices(self):
        return self.recommendations_indices

    def get_values(self):
        return self.recommendations_values

    def get_recommendations_count(self):
        return len(self.recommendations_values)


