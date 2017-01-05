#!/usr/bin/env python
import numpy
from lib.content_based import ContentBased
from sklearn.feature_extraction.text import CountVectorizer

import chainer
from chainer import optimizers
from lda2vec import preprocess
from lda2vec import utils
from lda2vec_model import LDA2Vec


class LDA2VecRecommender(ContentBased):
    def __init__(self, abstracts, evaluator, config, verbose=False):
        super(LDA2VecRecommender, self).__init__(abstracts, evaluator, config, verbose)

    def train(self, n_iter=5):
        """
        Train the LDA2Vec model.
        """
        skip_character = -2
        n_units = max(map(lambda doc: len(doc.split(' ')), self.abstracts)) + 1
        arr, vocab = preprocess.tokenize(map(unicode, self.abstracts), n_units, skip=skip_character)
        doc_ids = []
        flattened = []
        for doc_id, words in enumerate(arr):
            for word_id in words:
                if word_id != skip_character:
                    doc_ids.append(doc_id)
                    flattened.append(word_id)
        flattened = numpy.array(flattened)
        doc_ids = numpy.array(doc_ids)
        n_vocab = flattened.max() + 1
        tok_idx, freq = numpy.unique(flattened, return_counts=True)
        term_frequency = numpy.zeros(n_vocab, dtype='int32')
        term_frequency[tok_idx] = freq

        assert doc_ids.max() + 1 == self.n_items
        print(self.n_items, self.n_factors, n_units, n_vocab,
              map(lambda y: (y, vocab[y[0]]), filter(lambda x: x[1] != 0, enumerate(term_frequency))))
        lda2v_model = LDA2Vec(n_documents=self.n_items, n_document_topics=self.n_factors,
                              n_units=n_units, n_vocab=n_vocab, counts=term_frequency)

        optimizer = optimizers.Adam()
        optimizer.setup(lda2v_model)
        clip = chainer.optimizer.GradientClipping(5.0)
        optimizer.add_hook(clip)
        it = 0
        for epoch in range(n_iter):
            optimizer.zero_grads()
            l = lda2v_model.fit_partial(doc_ids.copy(), flattened.copy())
            prior = lda2v_model.prior()
            loss = prior
            loss.backward()
            optimizer.update()
            if self._v:
                msg = ("IT:{it:05d} E:{epoch:05d} L:{loss:1.3e} P:{prior:1.3e}")
                logs = dict(loss=float(l), epoch=epoch, it=it, prior=float(prior.data))
                print(msg.format(**logs))
            it += 1
        # self.word_distribution = lda.fit_transform(term_freq)

    def split(self):
        return super(LDA2VecRecommender, self).split()

    def set_config(self, config):
        """
        set the hyperparamenters of the algorithm.
        """
        super(LDA2VecRecommender, self).set_config(config)

    def get_word_distribution(self):
        """
        @returns a matrix of the words x topics distribution.
        """
        return self.word_distribution
