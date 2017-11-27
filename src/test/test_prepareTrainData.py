# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader import DataLoader
from data.prepare_train_data import PrepareTrainData
import os
import pickle
import filecmp


class TestPrepareTrainData(TestCase):
    def test_prepare_train_data_method(self):
        self.test_data_loader = DataLoader()
        self.test_japanese_wiki_data = 'test/test_data/jawiki_test.txt'
        test_word2index, test_index2word, test_window_data, \
            test_X_ik, test_weightinhg_dict = self.test_data_loader.load_data(file_name=self.test_japanese_wiki_data)  # noqa
        self.test_prepare_train_data = PrepareTrainData()
        test_train_data = \
            self.test_prepare_train_data.prepare_train_data_method(
                window_data=test_window_data,
                word2index=test_word2index,
                weighting_dic=test_weightinhg_dict,
                X_ik=test_X_ik)
        APP_PATH = os.path.dirname(__file__)
        output_file = APP_PATH + '/test_data/train_data.pkl'
        compare_output_file = APP_PATH + '/test_data/test_train_data.pkl'
        with open(output_file, 'wb') as handle:
            pickle.dump(test_train_data, handle)
        assert True is filecmp.cmp(output_file, compare_output_file)
