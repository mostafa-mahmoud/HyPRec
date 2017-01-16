#!/usr/bin/env python
"""
Module to preprocess abstracts.
"""
import numpy
from scipy import sparse


class AbstractsPreprocessor(object):
    """
    A class that computes necessary preprocessing for abstracts and query them.
    """
    def __init__(self, abstracts, word_to_count, article_to_word, article_to_word_to_count):
        """
        Constructs an abstracts preprocessor.

        :param dict[int,str] abstracts: List of the abstracts of the documents.
        :param list[pair] word_to_count: List of (word_id, word_count) pairs.
        :param list[pair] article_to_word: List of (article_id, word_id) for all words in abstracts.
        :param list[triple] article_to_word_to_count:
            List of (article_id, word_id, count) for the count of each word in all the abstracts.
        """
        self.abstracts = abstracts
        self.word_to_count = word_to_count
        self.article_to_word = article_to_word
        self.article_to_word_to_count = article_to_word_to_count

    def get_abstracts(self):
        """
        :returns: List of abstracts.
        :rtype: list[str]
        """
        abstracts = ['' for _ in range(self.get_num_items())]
        for doc_id, abstract in self.abstracts.items():
            abstracts[doc_id] = abstract
        return abstracts

    def get_word_to_counts(self):
        """
        :returns: List of (word_id, word_count) pairs.
        :rtype: list[pair[int]]
        """
        return self.word_to_count

    def get_article_to_words(self):
        """
        :returns: List of (article_id, word_id) for all words in abstracts.
        :rtype: list[pair[int]]
        """
        return self.article_to_word

    def get_article_to_word_to_count(self):
        """
        :returns: List of (article_id, word_id, count) for the count of each word in all the abstracts.
        :rtype: list[triple[int]]
        """
        return self.article_to_word_to_count

    def get_term_frequency_sparse_matrix(self):
        """
        :returns: Sparse matrix of documents X words, of the word count.
        :rtype: csr_matrix
        """
        articles, words, counts = zip(*self.get_article_to_word_to_count())
        return sparse.coo_matrix((counts, (articles, words)),
                                 shape=(self.get_num_items(), self.get_num_vocab())).tocsr()

    def get_num_vocab(self):
        """
        :returns: The size of the vocabulary.
        :rtype: int
        """
        return max(map(lambda inp: inp[0], self.word_to_count)) + 1

    def get_num_items(self):
        """
        :returns: The number of items given.
        :rtype: int
        """
        return max(self.abstracts.keys()) + 1

    def get_term_frequencies(self):
        """
        :returns: The list of frequencies of words.
        :rtype: list
        """
        counts = [0 for _ in range(self.get_num_vocab())]
        for word_id, word_count in self.get_word_to_counts():
            counts[word_id] = word_count
        return numpy.array(counts)

    def get_num_units(self):
        """
        :returns: The (maximum) number of words in each article.
        :rtype: int
        """
        return max(map(lambda doc: len(doc.split(' ')), self.abstracts.values()))
