# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from models.trainer import Trainer
import argparse
from data.data_loader_attention import DataLoaderAttention
from models.qrnn_model import QRNNModel
from models.qrnn_layer import QRNNLayer


def main():
    parser = argparse.ArgumentParser(
        description="Training attention model")

    parser.add_argument("-t", "--train_data", metavar="train_data",
                        type=str, default='../data/processed/source_replay_twitter_data.txt',
                        dest="train_data", help="set the training data ")
    parser.add_argument("-e", "--embedding_size", metavar="embedding_size",
                        type=int, default=50,
                        dest="embedding_size", help="set the embedding size ")
    parser.add_argument("-H", "--hidden_size", metavar="hidden_size",
                        type=int, default=512,
                        dest="hidden_size", help="set the hidden size ")
    parser.add_argument("-f", "--fine_tune_model_name", metavar="fine_tune_model_name",
                        type=str, default='../models/glove_model_40.pth',
                        dest="fine_tune_model_name", help="set the fine tune model name ")
    args = parser.parse_args()

    data_loader_attention = DataLoaderAttention(file_name=args.train_data)
    data_loader_attention.load_data()
    source2index, index2source, target2index, index2target, train_data = \
        data_loader_attention.load_data()
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = 2
    KERNEL_SIZE = 2
    EMBEDDING_SIZE = args.embedding_size
    SOURCE_VOCAB_SIZE = len(source2index)
    TARGET_VOCAB_SIZE = len(target2index)

    qrnn = QRNNModel(QRNNLayer, NUM_LAYERS, KERNEL_SIZE, HIDDEN_SIZE,
                     EMBEDDING_SIZE, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE)

    trainer = Trainer(
        epoch=300, batch_size=64,
        fine_tune_model=args.fine_tune_model_name
    )

    trainer.train_qrnn(train_data=train_data,
                       source2index=source2index,
                       target2index=target2index,
                       index2source=index2source,
                       index2target=index2target,
                       qrnn_model=qrnn,
                       )


if __name__ == "__main__":
    main()
