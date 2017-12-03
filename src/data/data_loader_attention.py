# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from utils import read_file
from utils import prepare_sequence
import unicodedata
import re
import collections
flatten = lambda l: [item for sublist in l for item in sublist]  # noqa


class DataLoaderAttention(object):

    def __init__(self,
                 file_name: str='',
                 MIN_LENGTH: int=3,
                 MAX_LENGTH: int=25,
                 ):
        self.file_name = file_name
        self.MIN_LENGTH = MIN_LENGTH
        self.MAX_LENGTH = MAX_LENGTH

    def load_data(self):
        source2index, index2source, target2index, index2target, train_data = \
            self.__read_data()
        return source2index, index2source, target2index, index2target, train_data

    def __read_data(self):
        read_data = read_file(file_name=self.file_name)
        X_r, y_r = [], []
        for parallel in read_data:
            if len(parallel[:-1].split('\t')) < 2:
                continue
            source, target = parallel[:-1].split('\t')
            if source.strip() == '' or target.strip() == '':
                continue
            normalize_source = self.__normalize_string(source).split()
            normalize_target = self.__normalize_string(target).split()

            if len(normalize_source) >= self.MIN_LENGTH and len(normalize_source) <= self.MAX_LENGTH \
                and len(normalize_target) >= self.MIN_LENGTH and len(normalize_target) <= self.MAX_LENGTH:
                X_r.append(normalize_source)
                y_r.append(normalize_target)
        source2index, index2source, target2index, index2target = self.__build_vocab(X_r=X_r,
                                                                                    y_r=y_r)
        train_data = self.__prepare_train_data(X_r=X_r,
                                               y_r=y_r,
                                               source2index=source2index,
                                               target2index=target2index)
        return source2index, index2source, target2index, index2target, train_data

    def __build_vocab(self, X_r: list, y_r: list):
        source_vocab = sorted(list(set(flatten(X_r))))
        target_vocab = sorted(list(set(flatten(y_r))))
        source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
        for vo in source_vocab:
            if vo not in source2index.keys():
                source2index[vo] = len(source2index)
        index2source = {v:k for k, v in source2index.items()}
        target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
        for vo in target_vocab:
            if vo not in target2index.keys():
                target2index[vo] = len(target2index)
        index2target = {v:k for k, v in target2index.items()}
        return source2index, index2source, target2index, index2target

    def __prepare_train_data(self, X_r: list, y_r: list,
                             source2index: list,
                             target2index: list):
        X_p, y_p = [], []
        for source, target in zip(X_r, y_r):
            X_p.append(prepare_sequence(['<s>'] + source + ['</s>'], source2index).view(1, -1))
            y_p.append(prepare_sequence(['<s>'] + target + ['</s>'], target2index).view(1, -1))

        train_data = list(zip(X_p, y_p))
        return train_data

    def __unicode_to_ascii(self, string):
        return ''.join(c for c in unicodedata.normalize('NFD', string)
                       if unicodedata.category(c) != 'Mn')

    def __normalize_string(self, string: str):
        string = self.__unicode_to_ascii(string.lower().strip())
        string = re.sub(r'([,!?])', r' \1 ', string)
        # string = re.sub(r'[^a-zA-Z,.!?]+', r' ', string)
        string = re.sub(r'\s+', r' ', string).strip()
        return string
