# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from visualization.attention_visualize import AttentionVisualize
from data.data_loader_attention import DataLoaderAttention
from utils import USE_CUDA
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_data", metavar="train_data",
                        type=str, default='../data/processed/source_replay_twitter_data.txt',
                        dest="train_data", help="set the training data ")
    args = parser.parse_args()
    test_data_loader_attention = DataLoaderAttention(file_name=args.train_data)
    test_data_loader_attention.load_data()
    source2index, index2source, target2index, index2target, train_data = \
        test_data_loader_attention.load_data()


    encoder_model_name = '../models/encoder_model_299.pth'
    decoder_model_name = '../models/decoder_model_299.pth'
    attention_visualize = AttentionVisualize(encoder_model_name=encoder_model_name,
                                             decoder_model_name=decoder_model_name)

    test = random.choice(train_data)
    inputs = test[0]
    truth = test[1]

    output, hidden = attention_visualize.encoder_model(inputs, [inputs.size(1)])
    pred, atten = attention_visualize.decoder_model.decode(hidden, output, target2index, index2target)

    inputs = [index2source[i] for i in inputs.data.tolist()[0]]
    pred = [index2target[i] for i in pred.data.tolist()]

    print('Source : ', ' '.join([i for i in inputs if i not in ['</s>']]))
    print('Truth : ', ' '.join([index2target[i] for i in truth.data.tolist()[0] if i not in [2, 3]]))
    print('Prediction : ', ' '.join([i for i in pred if i not in ['</s>']]))

    if USE_CUDA:
        atten = atten.cpu()

    attention_visualize.visualize(inputs, pred, atten.data)

if __name__ == "__main__":
    main()

