# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader_attention import DataLoaderAttention
from models.encoder import Encoder
from models.decoder import Decoder
from models.trainer import Trainer


class TestTrainerAttention(TestCase):
    def test_train_method(self):
        file_name = 'test/test_data/attention_test.txt'
        fine_tune_model_name = '../models/glove_model_40.pth'
        self.test_data_loader_attention = DataLoaderAttention(file_name=file_name)
        self.test_data_loader_attention.load_data()
        source2index, index2source, target2index, index2target, train_data = \
            self.test_data_loader_attention.load_data()
        EMBEDDING_SIZE = 50
        HIDDEN_SIZE = 32

        encoder = Encoder(len(source2index), EMBEDDING_SIZE, HIDDEN_SIZE, 3, True)
        decoder = Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE*2)

        self.trainer = Trainer(
            fine_tune_model=fine_tune_model_name
        )

        self.trainer.train_attention(train_data=train_data,
                                     source2index=source2index,
                                     target2index=target2index,
                                     index2source=index2source,
                                     index2target=index2target,
                                     encoder_model=encoder,
                                     decoder_model=decoder,
                                     )

