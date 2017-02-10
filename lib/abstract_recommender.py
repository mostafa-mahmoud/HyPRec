#!/usr/bin/env python
"""
This is a module that contains an abstract class AbstractRecommender.
"""
import numpy
from util.top_recommendations import TopRecommendations


class AbstractRecommender(object):
    """
    A class that acts like an interface, it is never initialized but the uv_decomposition
    and content_based should implement it's methods.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options, **flags):
        raise NotImplementedError("Can't initialize this class")

    def __repr__(self):
        return self.__class__.__name__

    def set_options(self, options):
        """
        Set the options of the recommender. Namely n_iterations and k_folds.

        :param dict options: A dictionary of the options.
        """
        self.n_iter = options['n_iterations']
        self.k_folds = options['k_folds']
        self.splitting_method = 'kfold'
        self.evaluator.set_kfolds(self.k_folds)

        if self.k_folds == 1:
            self.splitting_method = 'naive'
        self.options = options.copy()

    def set_hyperparameters(self, hyperparameters):
        """
        The function sets the hyperparameters of the recommender.

        :param dict hyperparameters: hyperparameters of the recommender.
        """
        raise NotImplementedError("Can't call this method")

    def predict(self, user, item):
        raise NotImplementedError("Can't call this method")

    def train_one_fold(self):
        raise NotImplementedError("Can't call this method")

    def train(self):
        """
        Train the content-based.
        """
        if self.splitting_method == 'naive':
            self.train_data, self.test_data = self.evaluator.naive_split()
            return self.train_one_fold()
        else:
            self.fold_train_indices, self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold()

    def train_k_fold(self):
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_train_indices,
                                                                      self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            self.train_one_fold()
            all_errors.append(self.get_evaluation_report())
        return numpy.mean(all_errors, axis=0)

    def get_evaluation_report(self):
        """
        Method prints evaluation report for a trained model.

        :returns: Tuple of evaluation metrics.
        :rtype: Tuple
        """
        predictions = self.get_predictions()
        rounded_predictions = self.rounded_predictions()
        test_sum = self.test_data.sum()
        train_sum = self.train_data.sum()
        self.evaluator.load_top_recommendations(200, predictions, self.test_data)
        train_recall = self.evaluator.calculate_recall(self.train_data, rounded_predictions)
        test_recall = self.evaluator.calculate_recall(self.test_data, rounded_predictions)
        recall_at_x = self.evaluator.recall_at_x(200, predictions, self.test_data, rounded_predictions)
        recommendations = sum(sum(rounded_predictions))
        likes = sum(sum(self.ratings))
        ratio = recommendations / likes
        mrr_at_five = self.evaluator.calculate_mrr(5, predictions, self.test_data, rounded_predictions)
        ndcg_at_five = self.evaluator.calculate_ndcg(5, predictions, self.test_data, rounded_predictions)
        mrr_at_ten = self.evaluator.calculate_mrr(10, predictions, self.test_data, rounded_predictions)
        ndcg_at_ten = self.evaluator.calculate_ndcg(10, predictions, self.test_data, rounded_predictions)
        rmse = self.evaluator.get_rmse(predictions, self.ratings)
        if self._verbose:
            report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.3f}, train recall {:.3f}, '\
                         'test recall {:.3f}, recall@200 {:.3f}, '\
                         'ratio {:.3f}, mrr@5 {:.3f}, '\
                         'ndcg@5 {:.3f}, mrr@10 {:.3f}, ndcg@10 {:.3f}'
            print(report_str.format(test_sum, train_sum, rmse, train_recall, test_recall, recall_at_x, ratio,
                                    mrr_at_five, ndcg_at_five, mrr_at_ten, ndcg_at_ten))
        return (rmse, train_recall, test_recall, recall_at_x, ratio, mrr_at_five, ndcg_at_five,
                mrr_at_ten, ndcg_at_ten)

    def get_predictions(self):
        """
        Get the predictions matrix. Initialized properly after calling 'train'

        :returns: A userXdocument matrix of predictions
        :rtype: ndarray
        """
        return self.predictions

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
