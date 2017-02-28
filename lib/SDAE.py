#!/usr/bin/env python
"""
A module that contains the content-based recommender SDAERecommender that
stacked denoising autoencoders.
"""
import time
import numpy
from overrides import overrides
from keras import backend
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation
from keras.layers.core import Dense, Reshape
from keras.models import Model
from keras.models import Sequential
from lda2vec.utils import chunks
from lib.content_based import ContentBased


class SDAERecommender(ContentBased):
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
        super(SDAERecommender, self).__init__(initializer, evaluator, hyperparameters, options,
                                              verbose, load_matrices, dump_matrices)

    @overrides
    def train_one_fold(self):
        """
        Try to load saved matrix if load_matrices is false, else train
        """
        matrix_found = False
        if self._load_matrices is True:
            matrix_shape = (self.n_items, self.n_factors)
            matrix_found, matrix = self.initializer.load_matrix(self.hyperparameters,
                                                                'document_distribution_sdae', matrix_shape)
            self.document_distribution = matrix
            if self._verbose and matrix_found:
                print("Document distribution was set from file, will not train.")
        if matrix_found is False:
            if self._verbose and self._load_matrices:
                print("Document distribution file was not found. Will train SDAE.")
            self._train()
            if self._dump_matrices:
                self.initializer.save_matrix(self.document_distribution, 'document_distribution_sdae')

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

    def _train(self):
        """
        Train the stacked denoising autoencoders.
        """
        encode_cnn, cnn = self.get_cnn()
        if self._verbose:
            "CNN is constructed..."
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix().todense()
        rand_term_freq = numpy.random.normal(term_freq, 0.25)
        iterations = 0
        batchsize = 2048
        for epoch in range(1, 1 + self.n_iter):
            for inp_batch, out_batch in chunks(batchsize, rand_term_freq, term_freq):
                t0 = time.time()
                l = cnn.train_on_batch(inp_batch, out_batch)
                t1 = time.time()
                iterations += 1
                if self._verbose:
                    msg = "Iteration:{it:05d} Epoch:{epoch:02d} Loss:{loss:1.3e} Time:{tim:.3f}s"
                    logs = dict(loss=float(l), epoch=epoch, it=iterations, tim=(t1 - t0))
                    print(msg.format(**logs))

        self.document_distribution = encode_cnn.predict(term_freq)
        rms = cnn.evaluate(term_freq, term_freq)

        if self._verbose:
            print(rms)
        # Garbage collection for keras
        backend.clear_session()
        if self._verbose:
            print("SDAE trained...")
        return rms
