#!/usr/bin/env python


class AbstractRecommender(object):
    """
    A class that acts like an interface, it is never initialized but the uv_decomposition
    and content_based should implement it's methods.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options, **flags):
        raise NotImplementedError("Can't initialize this class")

    def train(self):
        assert self.n_iter is not None
        raise NotImplementedError("Can't call this method")

    def set_hyperparameters(self, hyperparameters):
        raise NotImplementedError("Can't call this method")

    def set_options(self, options):
        raise NotImplementedError("Can't call this method")

    def get_predictions(self):
        raise NotImplementedError("Can't call this method")

    def predict(self, user, item):
        raise NotImplementedError("Can't call this method")

    def get_ratings(self):
        """
        Getter for the ratings

        :returns: Ratings matrix
        :rtype: ndarray
        """
        return self.ratings

    def rounded_predictions(self):
        """
        The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.

        :returns: predictions rounded up matrix
        :rtype: int[][]
        """
        predictions = self.get_predictions()
        n_users = self.ratings.shape[0]
        for user in range(n_users):
            avg = sum(self.ratings[0]) / self.ratings.shape[1]
            low_values_indices = predictions[user, :] < avg
            predictions[user, :] = 1
            predictions[user, low_values_indices] = 0
        return predictions

    def recommend_items(self, user_id, num_recommendations=10):
        """
        Get recommendations for a user. Based on the predictions returned by get_predictions

        :param int user_id: The id of the user.
        :param int num_recommendations: The number of recommended items.
        :returns:
            A zipped object containing list of tuples; first index is the id of the document
            and the second is the value of the calculated recommendation.
        :rtype: zip
        """
        top_recommendations = TopRecommendations(num_recommendations)
        user_ratings = self.get_predictions()[user_id]
        for i in range(len(user_ratings)):
            top_recommendations.insert(i, user_ratings[i])
        return zip(top_recommendations.get_indices(), top_recommendations.get_values())
