# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import nltk
import collections
from itertools import combinations_with_replacement
from collections import Counter
from utils import read_file
flatten = lambda l: [item for sublist in l for item in sublist]  # noqa


class DataLoader(object):

    def __init__(self):
        self.vocab = []

    def load_data(self, file_name: str=''):
        self.file_name = file_name
        word2index, index2word, window_data, X_ik, weightinhg_dict = \
            self.__read_data()
        return word2index, index2word, window_data, X_ik, weightinhg_dict

    def __read_data(self):
        corpus = read_file(self.file_name)
        corpus = [[word.lower() for word in sent] for sent in corpus]
        word2index, index2word, vocab = self.__build_vocab(corpus=corpus)
        window_data = self.__make_window_data(corpus=corpus)
        X_ik, weightinhg_dict = self.__make_co_occurence_matrix(
            window_data=window_data,
            vocab=vocab)
        return word2index, index2word, window_data, X_ik, weightinhg_dict

    def __build_vocab(self, corpus: list=[]):
        self.vocab = sorted(list(set(flatten(corpus))))
        word2index, index2word = self.__make_word2index(vocab=self.vocab)
        return word2index, index2word, self.vocab

    def __make_word2index(self, vocab: list=[]):
        word2index = {}
        for vo in vocab:
            if vo not in word2index.keys():
                word2index[vo] = len(word2index)
        index2word = {v: k for k, v in word2index.items()}
        word2index = dict(collections.OrderedDict(sorted(word2index.items(),
                                                         key=lambda t: t[1])))
        index2word = dict(collections.OrderedDict(sorted(index2word.items(),
                                                         key=lambda t: t[0])))
        return word2index, index2word

    def __make_window_data(self, window_size: int=5,
                           corpus: list=[]):
        windows = flatten([list(nltk.ngrams(['<DUMMY>'] * window_size + c +
                                            ['<DUMMY>'] * window_size,
                                            window_size*2+1)) for c in corpus])
        window_data = []
        for window in windows:
            for i in range(window_size*2 + 1):
                if i == window_size or window[i] == '<DUMMY>':
                    continue
                window_data.append((window[window_size], window[i]))
        return window_data

    def __weighting(self, X_ik: list, w_i: int, w_j: int):
        try:
            x_ij = X_ik[(w_i, w_j)]
        except:
            x_ij = 1

        x_max = 100
        alpha = 0.75

        if x_ij < x_max:
            result = (x_ij / x_max) ** alpha
        else:
            result = 1
        return result

    def __make_co_occurence_matrix(self,
                                   window_data: list=[],
                                   vocab: list=[]):
        X_ik_window_5 = Counter(window_data)
        X_ik = {}
        weightinhg_dict = {}
        for bigram in combinations_with_replacement(vocab, 2):
            if bigram in X_ik_window_5.keys():
                co_occer = X_ik_window_5[bigram]
                X_ik[bigram] = co_occer + 1
                X_ik[bigram[1], bigram[0]] = co_occer + 1
            else:
                pass
            weightinhg_dict[bigram] = self.__weighting(X_ik=X_ik,
                                                       w_i=bigram[0],
                                                       w_j=bigram[1])
            weightinhg_dict[bigram[1], bigram[0]] = \
                self.__weighting(X_ik=X_ik, w_i=bigram[1], w_j=bigram[0])
        weightinhg_dict = dict(collections.OrderedDict(
            sorted(weightinhg_dict.items(), key=lambda t: t[1])))
        return X_ik, weightinhg_dict
