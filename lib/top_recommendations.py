

class TopRecommendations(object):

    def __init__(self, n_recommendations):
        self.recommendations = []
        self.current_recommendations = 0
        self.lowest_recommendation = 0
        self.lowest_recommendation_index = None
        self.n_recommendations = n_recommendations

    def insert(self, index, value):
        # always append if the array didn't hit the size
        if self.lowest_recommendation_index is None:
            self.lowest_recommendation_index = 0
        if self.n_recommendations > self.current_recommendations + 1:
            self.recommendations.append((index, value))
            self.current_recommendations += 1
            if self.lowest_recommendation > value:
                self.lowest_recommendation = value
                self.lowest_recommendation_index = self.current_recommendations - 1
        # check if the newly appended is the biggest
        elif value > self.lowest_recommendation:
            self.recommendations[self.lowest_recommendation_index] = (index, value)
            self.set_lowest()

    def set_lowest(self):
        lowest_index = 0
        lowest = self.recommendations[lowest_index][1]
        for i in range(len(self.recommendations)):
            current_recommendation = self.recommendations[i][1]
            if lowest > current_recommendation:
                lowest = current_recommendation
                lowest_index = i
        self.lowest_recommendation = lowest
        self.lowest_recommendation_index = lowest_index

    def get_indices(self):
        indices = []
        for index, _ in self.recommendations:
            indices.append(index)
        return indices

    def get_values(self):
        values = []
        for _, value in self.recommendations:
            values.append(value)
        return values

    def get_recommendations(self):
        return self.recommendations

    def get_recommendations_count(self):
        return self.current_recommendations - 1


