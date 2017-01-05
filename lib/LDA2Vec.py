#!/usr/bin/env python
"""
A module that contains the content-based recommender LDA2VecRecommender that
uses the LDA2Vec library.
"""
import chainer
import numpy
from chainer import optimizers
from lda2vec import preprocess
from lda2vec_model import LDA2Vec
from lib.content_based import ContentBased


class LDA2VecRecommender(ContentBased):
    """
    LDA2Vec recommender, a content based recommender that uses LDA2Vec.
    """
    def __init__(self, abstracts, evaluator, config, verbose=False):
        """
        Constructor of ContentBased processor.
        @param abstracts list(str): List of the texts of the abstracts of the papers.
        @param evaluator: An evaluator object.
        @param config(dict): A dictionary of the hyperparameters.
        @param verbose(boolean): A flag for printing while computing.
        """
        super(LDA2VecRecommender, self).__init__(abstracts, evaluator, config, verbose)

    def train(self, n_iter=5):
        """
        Train the LDA2Vec model, and store the document_distribution matrix.
        """
        # Tokenize words into 2d array of shape (documents, unit) of word_ids
        # and the translation vocabulary 'vocab'
        skip_character = -2
        n_units = max(map(lambda doc: len(doc.split(' ')), self.abstracts)) + 1
        arr, vocab = preprocess.tokenize(map(unicode, self.abstracts), n_units, skip=skip_character)

        # 2 lists which correspond to pairs ('doc_id', 'word_id') of all the words
        # in each document, 'word_id' according to the computed dictionary 'vocab'
        doc_ids = []
        flattened = []
        for doc_id, words in enumerate(arr):
            for word_id in words:
                if word_id != skip_character:
                    doc_ids.append(doc_id)
                    flattened.append(word_id)
        flattened = numpy.array(flattened, dtype='int32')
        doc_ids = numpy.array(doc_ids, dtype='int32')

        # Word frequencies, for lda2vec_model
        n_vocab = flattened.max() + 1
        tok_idx, freq = numpy.unique(flattened, return_counts=True)
        term_frequency = numpy.zeros(n_vocab, dtype='int32')
        term_frequency[tok_idx] = freq

        # Assuming that doc_ids are in the set {0, 1, ..., n - 1}
        assert doc_ids.max() + 1 == self.n_items
        if self._v:
            print(self.n_items, self.n_factors, n_units, n_vocab, len(vocab),
                  map(lambda y: (y, vocab[y[0]]), filter(lambda x: x[1] != 0, enumerate(term_frequency))))
        # Initialize lda2vec model
        lda2v_model = LDA2Vec(n_documents=self.n_items, n_document_topics=self.n_factors,
                              n_units=n_units, n_vocab=n_vocab, counts=term_frequency)

        # Initialize optimizers
        optimizer = optimizers.Adam()
        optimizer.setup(lda2v_model)
        clip = chainer.optimizer.GradientClipping(5.0)
        optimizer.add_hook(clip)
        iterations = 0
        for epoch in range(n_iter):
            optimizer.zero_grads()
            l = lda2v_model.fit_partial(doc_ids.copy(), flattened.copy())
            prior = lda2v_model.prior()
            loss = prior
            loss.backward()
            optimizer.update()
            if self._v:
                msg = ("IT:{it:05d} E:{epoch:05d} L:{loss:1.3e} P:{prior:1.3e}")
                logs = dict(loss=float(l), epoch=epoch, it=iterations, prior=float(prior.data))
                print(msg.format(**logs))
            iterations += 1

        # Get document distribution matrix.
        self.document_distribution = lda2v_model.mixture.proportions(numpy.unique(doc_ids), True).data

    def split(self):
        """
        split the data into train and test data.
        @returns (tuple) A tuple of (train_data, test_data)
        """
        return super(LDA2VecRecommender, self).split()

    def set_config(self, config):
        """
        set the hyperparamenters of the algorithm.
        @param config(dict): A dictionary of the hyperparameters.
        """
        super(LDA2VecRecommender, self).set_config(config)

    def get_document_topic_distribution(self):
        """
        @returns A matrix of documents X topics distribution.
        """
        return self.document_distribution
