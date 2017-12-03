# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.data_loader_attention import DataLoaderAttention
from models.qrnn_model import QRNNModel
from models.qrnn_layer import QRNNLayer
from models.trainer import Trainer


class TestTrainerQRNN(TestCase):
    def test_train_method(self):
        file_name = 'test/test_data/attention_test.txt'
        fine_tune_model_name = '../models/glove_model_40.pth'
        self.test_data_loader_attention = DataLoaderAttention(file_name=file_name)
        self.test_data_loader_attention.load_data()
        source2index, index2source, target2index, index2target, train_data = \
            self.test_data_loader_attention.load_data()
        HIDDEN_SIZE = 512
        NUM_LAYERS = 2
        KERNEL_SIZE = 2
        EMBEDDING_SIZE = 50
        SOURCE_VOCAB_SIZE = len(source2index)
        TARGET_VOCAB_SIZE = len(target2index)

        qrnn = QRNNModel(QRNNLayer, NUM_LAYERS, KERNEL_SIZE, HIDDEN_SIZE,
                         EMBEDDING_SIZE, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE)

        self.trainer = Trainer(
            epoch=100,
            fine_tune_model=fine_tune_model_name
        )

        self.trainer.train_qrnn(train_data=train_data,
                                source2index=source2index,
                                target2index=target2index,
                                index2source=index2source,
                                index2target=index2target,
                                qrnn_model=qrnn,
                                )

