# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from models.classifier import Classifier
import filecmp
from data.data_loader import DataLoader


class TestClassifier(TestCase):
    def test_classify(self):
        model_name = '../models/glove_wiki/glove_model_40.pth'
        output_file = 'test/test_data/glove_classify_model.pkl'
        compare_output_file = 'glove_classify_model.pkl'
        classifier = Classifier(model_name=model_name)
        classifier.classify()
        assert True is filecmp.cmp(output_file, compare_output_file)

    def test_classify_predict(self):
        self.test_data_loader = DataLoader()
        self.test_japanese_wiki_data = 'test/test_data/jawiki_test.txt'
        test_word2index, test_index2word, test_window_data, \
        test_X_ik, test_weightinhg_dict = self.test_data_loader.load_data(file_name=self.test_japanese_wiki_data)  # noqa
        model_name = '../models/glove_wiki/glove_model_40.pth'
        output_file = 'test/test_data/glove_classify_model.pkl'
        classifier = Classifier(model_name=model_name)
        print(test_word2index)
        classes = classifier.classify_predict(word='の', classify_model_name=output_file,
                                              word2index=test_word2index)
        assert 2 == classes
        classes = classifier.classify_predict(word='どうよ?', classify_model_name=output_file,
                                              word2index=test_word2index)
        assert 9999 == classes
