# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader import DataLoader
import pickle
import os


class TestDataLoader(TestCase):

    def test_load_data(self):
        word2index = {'/': 0, '<': 1, '>': 2, 's': 3, '、': 4, '。': 5,
                      'が': 6, 'た': 7, 'で': 8, 'に': 9, 'の': 10,
                      'は': 11, 'を': 12}
        index2word = {0: '/', 1: '<', 2: '>', 3: 's', 4: '、', 5: '。',
                      6: 'が', 7: 'た', 8: 'で', 9: 'に', 10: 'の',
                      11: 'は', 12: 'を'}
        window_data = [('<', '/'), ('<', 's'), ('<', '>'), ('/', '<'),
                       ('/', 's'), ('/', '>'), ('s', '<'), ('s', '/'),
                       ('s', '>'), ('>', '<'), ('>', '/'), ('>', 's')]
        X_ik = {('/', '<'): 2, ('<', '/'): 2, ('/', '>'): 2, ('>', '/'): 2,
                ('/', 's'): 2, ('s', '/'): 2, ('<', '>'): 2, ('>', '<'): 2,
                ('<', 's'): 2, ('s', '<'): 2, ('>', 's'): 2, ('s', '>'): 2}

        self.test_data_loader = DataLoader()
        self.test_japanese_wiki_data = 'test/test_data/jawiki_test.txt'
        test_word2index, test_index2word, test_window_data, \
        test_X_ik, test_weightinhg_dict = self.test_data_loader.load_data(file_name=self.test_japanese_wiki_data)  # noqa
        # Reference
        #     https://stackoverflow.com/questions/11026959/writing-a-dict-to-txt-file-and-reading-it-back  # noqa
        APP_PATH = os.path.dirname(__file__)
        with open(APP_PATH + '/test_data/test_weighting_dict.pkl', 'rb') as handle:  # noqa
            weighting_dict = pickle.loads(handle.read())
        assert word2index == test_word2index
        assert index2word == test_index2word
        assert window_data == test_window_data
        assert test_X_ik == X_ik
        assert test_weightinhg_dict == weighting_dict
