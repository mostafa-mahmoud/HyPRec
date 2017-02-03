#!/usr/bin/env python
"""
A module that contains the content-based recommender LDA2VecRecommender that
uses the LDA2Vec library.
"""
import time
import chainer
import numpy
from chainer import optimizers
from overrides import overrides
from lda2vec.utils import chunks
from lib.lda2vec_model import LDA2Vec
from lib.content_based import ContentBased


class LDA2VecRecommender(ContentBased):
    """
    LDA2Vec recommender, a content based recommender that uses LDA2Vec.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True):
        """
        Constructor of LDA2Vec processor.

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: An evaluator of recommender and holder of input.
        :param dict hyperparameters: A dictionary of the hyperparameters.
        :param dict options: A dictionary of the run options.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the output matrices.
        """
        super(LDA2VecRecommender, self).__init__(initializer, evaluator, hyperparameters, options,
                                                 verbose, load_matrices, dump_matrices)

    @overrides
    def train(self):
        """
        Try to load saved matrix if load_matrices is false, else train
        """
        matrix_found = False
        if self._load_matrices is True:
            matrix_shape = (self.n_items, self.n_factors)
            matrix_found, matrix = self.initializer.load_matrix(self.hyperparameters,
                                                                'document_distribution_lda2vec', matrix_shape)
            self.document_distribution = matrix
            if self._verbose and matrix_found:
                print("Document distribution was set from file, will not train.")
        if matrix_found is False:
            if self._verbose and self._load_matrices:
                print("Document distribution file was not found. Will train LDA2Vec.")
            self._train()

    def _train(self):
        """
        Train the LDA2Vec model, and store the document_distribution matrix.
        """
        n_units = self.abstracts_preprocessor.get_num_units()
        # 2 lists which correspond to pairs ('doc_id', 'word_id') of all the words
        # in each document, 'word_id' according to the computed dictionary 'vocab'
        doc_ids, flattened = zip(*self.abstracts_preprocessor.get_article_to_words())
        assert len(doc_ids) == len(flattened)
        flattened = numpy.array(flattened, dtype='int32')
        doc_ids = numpy.array(doc_ids, dtype='int32')

        # Word frequencies, for lda2vec_model
        n_vocab = self.abstracts_preprocessor.get_num_vocab()
        term_frequency = self.abstracts_preprocessor.get_term_frequencies()
        if self._verbose:
            print('...')
            print('term_freq:')
            for word_count in filter(lambda x: x[1] != 0, zip(range(len(term_frequency)), term_frequency)):
                print(word_count)
            print('ratings:')
            for rating in zip(list(doc_ids), list(flattened)):
                print(rating)
            print(len(doc_ids))
            print('...')

        # Assuming that doc_ids are in the set {0, 1, ..., n - 1}
        assert doc_ids.max() + 1 == self.n_items
        if self._verbose:
            print(self.n_items, self.n_factors, n_units, n_vocab)
        # Initialize lda2vec model
        lda2v_model = LDA2Vec(n_documents=self.n_items, n_document_topics=self.n_factors,
                              n_units=n_units, n_vocab=n_vocab, counts=term_frequency)
        if self._verbose:
            print("Initialize LDA2Vec model..., Training LDA2Vec...")

        # Initialize optimizers
        optimizer = optimizers.Adam()
        optimizer.setup(lda2v_model)
        clip = chainer.optimizer.GradientClipping(5.0)
        optimizer.add_hook(clip)

        if self._verbose:
            print("Optimizer Initialized...")
        batchsize = 2048
        iterations = 0
        for epoch in range(1, self.n_iter + 1):
            for d, f in chunks(batchsize, doc_ids, flattened):
                t0 = time.time()
                if len(d) <= 10:
                    continue
                optimizer.zero_grads()
                l = lda2v_model.fit_partial(d.copy(), f.copy())
                prior = lda2v_model.prior()
                loss = prior
                loss.backward()
                optimizer.update()
                iterations += 1
                t1 = time.time()
                if self._verbose:
                    msg = "IT:{it:05d} E:{epoch:05d} L:{loss:1.3e} P:{prior:1.3e} T:{tim:.3f}s"
                    logs = dict(loss=float(l), epoch=epoch, it=iterations, prior=float(prior.data), tim=(t1 - t0))
                    print(msg.format(**logs))

        # Get document distribution matrix.
        self.document_distribution = lda2v_model.mixture.proportions(numpy.unique(doc_ids), True).data
        if self._dump_matrices:
            self.initializer.save_matrix(self.document_distribution, 'document_distribution_lda2vec')
        if self._verbose:
            print("LDA2Vec trained...")
