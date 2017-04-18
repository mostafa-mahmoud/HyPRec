#!/usr/bin/env python
"""
A module that contains the content-based recommender SDAERecommender that
stacked denoising autoencoders.
"""
import time
import numpy
from numpy.linalg import solve
from overrides import overrides
from keras import backend
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation
from keras.layers.core import Dense, Reshape
from keras.models import Model
from keras.models import Sequential
from lda2vec.utils import chunks
from lib.content_based import ContentBased
from lib.collaborative_filtering import CollaborativeFiltering


class SDAERecommender(CollaborativeFiltering, ContentBased):
    """
    Stacked denoising autoencoders, a content based recomender.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True):
        """
        Constructor of SDAE processor.

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: An evaluator of recommender and holder of input.
        :param dict hyperparameters: A dictionary of the hyperparameters.
        :param dict options: A dictionary of the run options.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the output matrices.
        """
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.get_ratings()
        self.abstracts_preprocessor = evaluator.get_abstracts_preprocessor()
        self.n_users, self.n_items = self.ratings.shape
        assert self.n_items == self.abstracts_preprocessor.get_num_items()
        self.prediction_fold = -1
        # setting flags
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._verbose = verbose
        self._update_with_items = True
        self._is_hybrid = False
        self._split_type = 'user'

        self.set_hyperparameters(hyperparameters)
        self.set_options(options)

    @overrides
    def train(self):
        """
        Train the SDAE.
        """
        self.document_distribution = None
        if self.splitting_method == 'naive':
            self.train_data, self.test_data = self.evaluator.naive_split(self._split_type)
            self.hyperparameters['fold'] = 0
            return self.train_one_fold()
        else:
            self.fold_train_indices, self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold()

    @overrides
    def train_k_fold(self):
        """
        Trains the k folds of SDAE.
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_train_indices,
                                                                      self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            current_error = self.train_one_fold()
            all_errors.append(current_error)
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        Set the  of the algorithm. Namely n_factors, _lambda.

        :param dict hyperparameters: A dictionary of the hyperparameters.
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.hyperparameters = hyperparameters.copy()

    @overrides
    def train_k_fold(self):
        """
        Trains the k folds of SDAE.
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_train_indices,
                                                                      self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            current_error = self.train_one_fold()
            all_errors.append(current_error)
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self):
        """
        Train model for n_iter iterations from scratch.
        """
        matrices_found = False
        if self._load_matrices is False:
            self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
            self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
        else:
            users_found, self.user_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'user_mat', (self.n_users, self.n_factors))
            if self._verbose and users_found:
                print("User distributions files were found.")
            items_found, self.item_vecs = self.initializer.load_matrix(self.hyperparameters, 'item_mat',
                                                                       (self.n_items, self.n_factors))
            if self._verbose and items_found:
                print("Document distributions files were found.")

            docs_found, self.document_distribution = self.initializer.load_matrix(self.hyperparameters,
                                                                                  'document_distribution_sdae',
                                                                                  (self.n_items, self.n_factors))
            if self._verbose and docs_found:
                print("Document latent distributions files were found.")
            matrices_found = users_found and items_found and docs_found

        if not matrices_found:
            if self._verbose and self._load_matrices:
                print("User and Document distributions files were not found, will train SDAE.")
            self._train()
        else:
            if self._verbose and self._load_matrices:
                print("User and Document distributions files found, will not train the model further.")

        if self._dump_matrices:
            self.initializer.set_config(self.hyperparameters, self.n_iter)
            self.initializer.save_matrix(self.user_vecs, 'user_mat')
            self.initializer.save_matrix(self.item_vecs, 'item_mat')
            self.initializer.save_matrix(self.document_distribution, 'document_distribution_sdae')

        return self.get_evaluation_report()

    def get_cnn(self):
        """
        Build a keras' convolutional neural network model.

        :returns: A tuple of 2 models, for encoding and encoding+decoding model.
        :rtype: tuple(Model)
        """
        model = Sequential()
        n_vocab = self.abstracts_preprocessor.get_num_vocab()
        n1, n2 = 64, 128
        model.add(Reshape((1, n_vocab,), input_shape=(n_vocab,)))
        model.add(Convolution1D(n1, 3, border_mode='same'))
        model.add(Activation('sigmoid'))
        model.add(Convolution1D(n2, 3, border_mode='same'))
        model.add(Activation('sigmoid'))
        model.add(Reshape((n2,)))
        model.add(Dense(n1))
        model.add(Activation('sigmoid'))
        model.add(Dense(n2))
        model.add(Reshape((1, n2)))
        model.add(Convolution1D(self.n_factors, 3, border_mode='same'))
        model.add(Activation('softmax'))
        model.add(Reshape((self.n_factors,), name='encoding'))
        intermediate = Model(input=model.input, output=model.get_layer('encoding').output)
        intermediate.compile(loss='mean_squared_error', optimizer='sgd')

        model.add(Reshape((1, self.n_factors)))
        model.add(Convolution1D(n2, 3, border_mode='same'))
        model.add(Activation('sigmoid'))
        model.add(Convolution1D(n1, 3, border_mode='same'))
        model.add(Activation('sigmoid'))
        model.add(Reshape((n1,)))
        model.add(Dense(n2))
        model.add(Activation('softmax'))
        model.add(Dense(n1))
        model.add(Activation('sigmoid'))
        model.add(Reshape((1, n1)))
        model.add(Convolution1D(n_vocab, 3, border_mode='same'))
        model.add(Reshape((n_vocab,)))

        model.compile(loss='mean_squared_error', optimizer='sgd')
        return intermediate, model

    def uv_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """
        The function computes only one step in the ALS algorithm

        :param ndarray latent_vectors: the vector to be optimized
        :param ndarray fixed_vecs: the vector to be fixed
        :param ndarray ratings: ratings that will be used to optimize latent * fixed
        :param float _lambda: reguralization parameter
        :param str type: either user or item.
        """
        if type == 'user':
            # Precompute
            lambdaI = numpy.eye(self.hyperparameters['n_factors']) * _lambda
            for u in range(latent_vectors.shape[0]):
                confidence = self.build_confidence_matrix(u, 'user')
                YTY = (fixed_vecs.T * confidence).dot(fixed_vecs)
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             (ratings[u, :] * confidence).dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            lambdaI = numpy.eye(self.hyperparameters['n_factors']) * _lambda
            for i in range(latent_vectors.shape[0]):
                confidence = self.build_confidence_matrix(i, 'item')
                XTX = (fixed_vecs.T * confidence).dot(fixed_vecs)
                if self.document_distribution is not None:
                    latent_vectors[i, :] = solve((XTX + lambdaI),
                                                 (ratings[:, i].T * confidence).dot(fixed_vecs) +
                                                 self.document_distribution[i, :] * _lambda)
                else:
                    latent_vectors[i, :] = solve((XTX + lambdaI), (ratings[:, i].T * confidence).dot(fixed_vecs))
        return latent_vectors

    def build_confidence_matrix(self, index, type='user'):
        """
        Builds a confidence matrix

        :param int index: Index of the user or item to build confidence for.
        :param str type: Type of confidence matrix, either user or item.

        :returns: A confidence matrix
        :rtype: ndarray
        """
        if type == 'user':
            shape = self.item_vecs.shape[0]
        else:
            shape = self.user_vecs.shape[0]

        confidence = numpy.array([0.1] * shape)
        for i in range(len(confidence)):
            if type == 'user':
                if self.train_data[index][i] == 1:
                    confidence[i] = 1
            else:
                if self.train_data[i][index] == 1:
                    confidence[i] = 1

        return confidence

    def _train(self):
        """
        Train the stacked denoising autoencoders.
        """
        if 'fold' in self.hyperparameters:
            current_fold = self.hyperparameters['fold'] + 1
        else:
            current_fold = 0
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix().todense()
        rand_term_freq = numpy.random.normal(term_freq, 0.25)
        encode_cnn, cnn = self.get_cnn()
        if self._verbose:
            print("CNN is constructed...")
        error = numpy.inf
        iterations = 0
        batchsize = 2048
        for epoch in range(1, 1 + self.n_iter):
            old_error = error
            self.document_distribution = encode_cnn.predict(term_freq)
            t0 = time.time()
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')

        iterations = 0
        batchsize = 2048
        for epoch in range(1, 1 + self.n_iter):
            t0 = time.time()
            self.document_distribution = encode_cnn.predict(term_freq)
            self.user_vecs = self.uv_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.uv_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')
            t1 = time.time()
            iterations += 1
            if self._verbose:
                error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
                if current_fold == 0:
                    logs = dict(it=iterations, epoch=epoch, loss=error, time=(t1 - t0))
                    print('Iteration:{it:05d} Epoch:{epoch:02d} Loss:{loss:1.4e} Time:{time:.3f}s'.format(**logs))
                else:
                    logs = dict(fold=current_fold, it=iterations, epoch=epoch, loss=error, time=(t1 - t0))
                    print('Fold:{fold:02d} Iteration:{it:05d} Epoch:{epoch:02d} Loss:{loss:1.4e} '
                          'Time:{time:.3f}s'.format(**logs))

            for inp_batch, out_batch, it_batch in chunks(batchsize, rand_term_freq, term_freq, self.item_vecs):
                t0 = time.time()
                l1 = encode_cnn.train_on_batch(inp_batch, it_batch)
                l2 = cnn.train_on_batch(inp_batch, out_batch)
                t1 = time.time()
                iterations += 1
                if self._verbose:
                    if current_fold == 0:
                        msg = ('Iteration:{it:05d} Epoch:{epoch:02d} LossR:{loss:1.3e} LossE:{lossid:1.3e} '
                               'Time:{tim:.3f}s')
                        logs = dict(loss=float(l2), lossid=float(l1), epoch=epoch, it=iterations, tim=(t1 - t0))
                        print(msg.format(**logs))
                    else:
                        msg = ('Fold:{fold:02d} Iteration:{it:05d} Epoch:{epoch:02d} '
                               'LossR:{loss:1.3e} LossE:{lossid:1.3e} Time:{tim:.3f}s')
                        logs = dict(fold=current_fold, loss=float(l2), lossid=float(l1), epoch=epoch,
                                    it=iterations, tim=(t1 - t0))
                        print(msg.format(**logs))
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if error >= old_error:
                if self._verbose:
                    print("Local Optimum was found in the last iteration, breaking.")
                break


        self.document_distribution = encode_cnn.predict(term_freq)
        rms = cnn.evaluate(term_freq, term_freq)

        if self._verbose:
            print(rms)
        # Garbage collection for keras
        backend.clear_session()
        if self._verbose:
            print("SDAE trained...")
        return rms

    @overrides
    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        return self.user_vecs.dot(self.item_vecs.T)

    @overrides
    def predict(self, user, item):
        """
        Single user and item prediction.

        :returns: prediction score
        :rtype: float
        """
        return self.user_vecs[user, :].dot(self.item_vecs[item, :].T)
