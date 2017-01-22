#!/usr/bin/env python
"""
This module saves and imports matrices.
"""
import numpy
import os


class ModelInitializer(object):
    """
    A class for importing and saving models.
    """
    def __init__(self, config, n_iterations, verbose=False):
        """
        Constructor for model initializer.

        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors.
        :param int n_iterations: Number of iterations used to train.
        """
        self.folder = 'matrices'
        self.set_config(config, n_iterations)
        self._v = verbose

    def set_config(self, config, n_iterations):
        """
        Function that sets config and n_iterations for the initializer.

        :param dict config: hyperparameters of the recommender, contains _lambda and n_factors.
        :param int n_iterations: Number of iterations used to train.
        """
        self.config = config
        self.config['n_iterations'] = n_iterations

    def save_matrix(self, matrix, matrix_name):
        """
        Function that dumps the matrix to a .dat file.

        :param ndarray matrix: Matrix to be dumped.
        :param str matrix_name: Name of the matrix to be dumped.
        """
        path = self._create_path(matrix_name, matrix.shape)
        matrix.dump(path)
        if self._v:
            print("dumped to %s" % path)

    def load_matrix(self, config, matrix_name, matrix_shape):
        """
        Function that loads a matrix from a file.

        :param dict config: Config that was used to calculate the matrix.
        :param str matrix_name: Name of the matrix to be loaded.
        :param tuple matrix_shape: A tuple of int containing matrix shape.
        :returns:
            A tuple of boolean (if the matrix is loaded or not)
            And the matrix if loaded, random matrix otherwise.
        :rtype: tuple
        """
        path = self._create_path(matrix_name, matrix_shape, config.copy())
        try:
            res = (True, numpy.load(path))
            if self._v:
                print("loaded from %s" % path)
            return res
        except FileNotFoundError:
            if self._v:
                print("File not found, %s will initialize randomly" % path)
            return (False, numpy.random.random(matrix_shape))

    def _generate_file_name(self, config, matrix_name):
        """
        Generate the file name from config and matrix_name.

        :param dict config: The hyperparameters config.
        :param str matrix_name: A string of the name of the matrix.
        :returns: A string representation of the file name.
        :rtype: str
        """
        keys_array = sorted(config)
        generated_key = str.join(',', ['%s-%s' % (key, str(config[key]).replace('.', '_')) for key in keys_array])
        return generated_key + matrix_name

    def _create_path(self, matrix_name, matrix_shape, config=None):
        """
        Function creates a string uniquely representing the matrix it also
        uses the config to generate the name.

        :param str matrix_name: Name of the matrix.
        :param int n_rows: Number of rows of the matrix.
        :returns: A string representing the matrix path.
        :rtype: str
        """
        n_rows = matrix_shape[0]
        if config is None:
            config = self.config
        if 'n_iterations' not in config.keys():
            config['n_iterations'] = self.config['n_iterations']
        config['n_factors'] = matrix_shape[1]
        config['n_rows'] = n_rows
        path = self._generate_file_name(config, matrix_name)
        base_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(base_dir), self.folder, path + '.dat')
