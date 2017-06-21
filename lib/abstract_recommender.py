#!/usr/bin/env python
"""
This is a module that contains an abstract class AbstractRecommender.
"""
import os
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
        Additionally, override all private options of recommender given in options,
        if we have an option {'name': value}, we will correspondingly will override self._name = value

        :param dict options: A dictionary of the options.
        """
        self.n_iter = options['n_iterations']
        self.k_folds = options['k_folds']
        self.splitting_method = 'kfold'
        self._split_type = 'user'
        self.evaluator.set_kfolds(self.k_folds)

        if self.k_folds == 1:
            self.splitting_method = 'naive'
        self.options = options.copy()

        for option, value in options.items():
            if hasattr(self, '_' + option):
                setattr(self, '_' + option, value)

    def set_hyperparameters(self, hyperparameters):
        """
        The function sets the hyperparameters of the recommender.

        :param dict hyperparameters: hyperparameters of the recommender.
        """
        raise NotImplementedError("Can't call this method")

    def predict(self, user, item):
        """
        Predict the rating of user on item.
        """
        raise NotImplementedError("Can't call this method")

    def train_one_fold(self):
        """
        Train one fold for n_iter iterations from scratch.
        """
        raise NotImplementedError("Can't call this method")

    def train(self):
        """
        Train the recommender
        """
        if self.splitting_method == 'naive':
            self.set_data(*self.evaluator.naive_split(self._split_type))
            return self.train_one_fold()
        else:
            self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold()

    def set_data(self, train_data, test_data):
        """
        Set the train and test data.

        :param int[][] train_data: Training data matrix
        :param int[][] test_data: Test data matrix
        """
        self.predictions = None
        self.train_data = train_data
        self.test_data = test_data

    def train_k_fold(self):
        """
        Trains k folds of the recommender.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.set_data(*self.evaluator.get_fold(current_k, self.fold_test_indices))
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
        self.evaluator.load_top_recommendations(200, predictions, self.test_data, self.hyperparameters['fold'])
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
        rmse = self.evaluator.get_rmse(predictions, self.train_data)
        if self._verbose:
            report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                         'test recall {:.5f}, recall@200 {:.5f}, '\
                         'ratio {:.5f}, mrr@5 {:.5f}, '\
                         'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
            print(report_str.format(test_sum, train_sum, rmse, train_recall, test_recall, recall_at_x, ratio,
                                    mrr_at_five, ndcg_at_five, mrr_at_ten, ndcg_at_ten))
        return (test_sum, train_sum, rmse, train_recall, test_recall, recall_at_x, ratio, mrr_at_five, ndcg_at_five,
                mrr_at_ten, ndcg_at_ten)

    def get_predictions(self):
        """
        Get the predictions matrix. Initialized properly after calling 'train'
        """
        raise NotImplementedError("Can't call this method")

    def get_ratings(self):
        """
        Getter for the ratings

        :returns: Ratings matrix
        :rtype: ndarray
        """
        return self.ratings

    def dump_recommendations(self, num_recommendations=10):
        """
        Dump the recommendations for all users.

        :param int num_recommendations: The number of recommendations for each user.
        """
        recommendations = []
        n_users = self.ratings.shape[0]
        for user in range(n_users):
            # Take only the 1-based id's of the non-zero ratings
            user_recommendations = map(lambda y: str(y[0] + 1),
                                       filter(lambda x: x[1] > 1e-6, self.recommend_items(user, num_recommendations)))
            recommendations.append(list(user_recommendations))
        base_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(os.path.dirname(base_dir), 'matrices/%s' % self.results_file_name)
        with open(path, "w") as f:
            for user_recommendations in recommendations:
                if user_recommendations:
                    f.write('%d %s\n' % (len(user_recommendations), str.join(' ', user_recommendations)))
                else:
                    f.write('%d\n' % len(user_recommendations))
            f.close()
        if self._verbose:
            print("dumped top recommendations to %s" % path)

    def rounded_predictions(self):
        """
        The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.

        :returns: predictions rounded up matrix
        :rtype: int[][]
        """
        predictions = self.get_predictions().copy()
        n_users = self.ratings.shape[0]
        for user in range(n_users):
            avg = sum(predictions[user]) / predictions.shape[1]
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
