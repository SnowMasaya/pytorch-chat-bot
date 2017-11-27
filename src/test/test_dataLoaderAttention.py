# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader_attention import DataLoaderAttention
import os
import pickle
import filecmp


class TestDataLoaderAttention(TestCase):
    def test_load_data(self):

        test_source2index = {'!': 4, '.': 5, '</s>': 3, '<PAD>': 0, '<UNK>': 1, '<s>': 2, 'co/otsehnz6dk': 6, 'https://t': 7, '歯': 8, '磨けよ': 9}
        test_index2source = {4: '!', 5: '.', 3: '</s>', 0: '<PAD>', 1: '<UNK>', 2: '<s>', 6: 'co/otsehnz6dk', 7: 'https://t', 8: '歯', 9: '磨けよ'}
        test_target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3, '.': 4, '?': 5, 'co/7jnltbaas': 6, 'https://t': 7, 'は': 8}
        test_index2target = {0: '<PAD>', 1: '<UNK>', 2: '<s>', 3: '</s>', 4: '.', 5: '?', 6: 'co/7jnltbaas', 7: 'https://t', 8: 'は'}
        file_name = 'test/test_data/attention_test.txt'
        self.test_data_loader_attention = DataLoaderAttention(file_name=file_name)
        self.test_data_loader_attention.load_data()
        source2index, index2source, target2index, index2target, train_data = \
            self.test_data_loader_attention.load_data()

        assert test_source2index == source2index
        assert test_index2source == index2source
        assert test_target2index == target2index
        assert test_index2target == index2target
        APP_PATH = os.path.dirname(__file__)
        output_file = APP_PATH + '/test_data/train_data_attention.pkl'
        compare_output_file = APP_PATH + '/test_data/test_train_data_attention.pkl'
        with open(output_file, 'wb') as handle:
            pickle.dump(train_data, handle)
        assert True is filecmp.cmp(output_file, compare_output_file)
