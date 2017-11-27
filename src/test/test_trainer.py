# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader import DataLoader
from data.prepare_train_data import PrepareTrainData
from models.glove import Glove
from models.trainer import Trainer


class TestTrainer(TestCase):
    def test_train_method(self):
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
        self.model = Glove(vocab_size=len(test_word2index))
        self.trainer = Trainer(model=self.model)

        self.trainer.train_method(train_data=test_train_data)

        word_similarity = self.trainer.word_similarity(
            target=self.test_data_loader.vocab[0],
            vocab=self.test_data_loader.vocab,
            word2index=test_word2index,
            top_rank=2
        )

        word_similarity_check = ['<', '>', 's']
        word_similarity_bool = False

        for word in word_similarity:
            if word[0] in word_similarity_check:
                word_similarity_bool = True

        assert word_similarity_bool is True

