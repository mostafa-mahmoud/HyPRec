#!/usr/bin/env python
"""
A module that contains the content-based recommender SDAERecommender that
stacked denoising autoencoders.
"""
import time
import numpy
from overrides import overrides
from keras import backend
from keras.layers import Convolution1D
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Reshape
from keras.models import Model
from keras.regularizers import l2
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

        :returns: List of error metrics.
        :rtype: list[float]
        """
        self.document_distribution = None
        if self.splitting_method == 'naive':
            self.train_data, self.test_data = self.evaluator.naive_split(self._split_type)
            self.hyperparameters['fold'] = 0
            return self.train_one_fold()
        else:
            self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold()

    @overrides
    def train_k_fold(self):
        """
        Trains the k folds of SDAE.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.train_data, self.test_data = self.evaluator.get_fold(current_k, self.fold_test_indices)
            self.hyperparameters['fold'] = current_k
            current_error = self.train_one_fold()
            all_errors.append(current_error)
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self):
        """
        Train model for n_iter iterations from scratch.

        :returns: List of error metrics.
        :rtype: list[float]
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
        n_vocab = self.abstracts_preprocessor.get_num_vocab()
        n1, n2 = 64, 128
        input_layer = Input(shape=(n_vocab,))
        model = Reshape((1, n_vocab,))(input_layer)
        model = Convolution1D(n1, 3, border_mode='same', activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Convolution1D(n2, 3, border_mode='same', activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Reshape((n2,))(model)
        model = Dense(n1, activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Dense(n2, W_regularizer=l2(.01))(model)
        model = Reshape((1, n2))(model)
        model = Convolution1D(self.n_factors, 3, border_mode='same',
                              activation='softmax', W_regularizer=l2(.01))(model)
        encoding = Reshape((self.n_factors,), name='encoding')(model)

        model = Reshape((1, self.n_factors))(encoding)
        model = Convolution1D(n2, 3, border_mode='same', activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Convolution1D(n1, 3, border_mode='same', activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Reshape((n1,))(model)
        model = Dense(n2, activation='softmax', W_regularizer=l2(.01))(model)
        model = Dense(n1, activation='sigmoid', W_regularizer=l2(.01))(model)
        model = Reshape((1, n1))(model)
        model = Convolution1D(n_vocab, 3, border_mode='same', W_regularizer=l2(.01))(model)
        decoding = Reshape((n_vocab,))(model)

        model = concatenate([encoding, decoding])
        self.model = Model(inputs=input_layer, outputs=model)
        self.model.compile(loss='mean_squared_error', optimizer='sgd')

    def train_sdae(self, X, y, std=0.25):
        """
        Train the stacked denoising autoencoders.

        :param ndarray X: input of the SDAE
        :param ndarray y: Target of the SDAE
        :param float std: The standard deviation of the noising of clean input.
        :returns: The loss of the training
        :rtype: float
        """
        return self.model.train_on_batch(X, numpy.concatenate((y, numpy.random.normal(X, std)), axis=1))

    def predict_sdae(self, X):
        """
        Predict the encoding of the stacked denoising autoencoders.

        :param ndarray X: input of the SDAE
        :param ndarray y: Target of the SDAE
        :returns: The encoded latent representation of X
        :rtype: float
        """
        encoded, decoded = numpy.split(self.model.predict(X), (-X.shape[1],), axis=1)
        return encoded

    def evaluate_sdae(self, X, y):
        """
        Compute the loss of the encoding of the stacked denoising autoencoders.

        :param ndarray X: input of the SDAE
        :param ndarray y: Target of the SDAE
        :returns: The encoded latent representation of X
        :rtype: ndarray
        """
        return self.model.evaluate(X, numpy.concatenate((y, X), axis=1))

    def _train(self):
        """
        Train the stacked denoising autoencoders.
        """
        if 'fold' in self.hyperparameters:
            current_fold = self.hyperparameters['fold'] + 1
        else:
            current_fold = 0
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix().todense()
        self.get_cnn()
        if self._verbose:
            print("CNN is constructed...")
        error = numpy.inf
        iterations = 0
        batchsize = 2048
        for epoch in range(1, 1 + self.n_iter):
            old_error = error
            self.document_distribution = self.predict_sdae(term_freq)
            t0 = time.time()
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')
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

            for inp_batch, item_batch in chunks(batchsize, term_freq, self.item_vecs):
                t0 = time.time()
                loss = self.train_sdae(inp_batch, item_batch)
                t1 = time.time()
                iterations += 1
                if self._verbose:
                    if current_fold == 0:
                        msg = ('Iteration:{it:05d} Epoch:{epoch:02d} Loss:{loss:1.3e} Time:{tim:.3f}s')
                        logs = dict(loss=float(loss), epoch=epoch, it=iterations, tim=(t1 - t0))
                        print(msg.format(**logs))
                    else:
                        msg = ('Fold:{fold:02d} Iteration:{it:05d} Epoch:{epoch:02d} Loss:{loss:1.3e} Time:{tim:.3f}s')
                        logs = dict(fold=current_fold, loss=float(loss), epoch=epoch, it=iterations, tim=(t1 - t0))
                        print(msg.format(**logs))
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if error >= old_error:
                if self._verbose:
                    print("Local Optimum was found in the last iteration, breaking.")
                break

        self.document_distribution = self.predict_sdae(term_freq)
        rms = self.evaluate_sdae(term_freq, self.item_vecs)

        if self._verbose:
            print(rms)
        # Garbage collection for keras
        backend.clear_session()
        if self._verbose:
            print("SDAE trained...")
        return rms
