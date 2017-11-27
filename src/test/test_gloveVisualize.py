# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from visualization.glove_visualize import GloveVisualize
from data.data_loader import DataLoader


class TestGloveVisualize(TestCase):
    def test_visualize(self):
        self.test_data_loader = DataLoader()
        self.test_japanese_wiki_data = '../data/raw/jawiki_only_word_random_choose.txt'
        test_word2index, test_index2word, test_window_data, \
        test_X_ik, test_weightinhg_dict = self.test_data_loader.load_data(
            file_name=self.test_japanese_wiki_data)  # noqa

        model_name = '../models/glove_wiki/glove_model_40.pth'

        self.test_glove_visualize = GloveVisualize(model_name=model_name)
        self.test_glove_visualize.visualize(vocab=self.test_data_loader.vocab)
