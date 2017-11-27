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
    parser.add_argument("-q", "--queue_size", metavar="queue_size",
                        type=int, default=30000,
                        dest="queue_size", help="set the queue size ")
    args = parser.parse_args()
    file_name = 'test/test_data/attention_test.txt'
    test_data_loader_attention = DataLoaderAttention(file_name=file_name)
    test_data_loader_attention.load_data()
    source2index, index2source, target2index, index2target, train_data = \
        test_data_loader_attention.load_data()


    encoder_model_name = '../models/encoder_model_40.pth'
    decoder_model_name = '../models/decoder_model_40.pth'
    attention_visualize = AttentionVisualize(encoder_model_name=encoder_model_name,
                                             decoder_model_name=decoder_model_name)

    test = random.choice(train_data)
    inputs = test[0]
    truth = test[1]

    output, hidden = attention_visualize.encoder_model(inputs, [inputs.size(1)])
    pred, atten = attention_visualize.decoder_model.decode(hidden, output, target2index)

    inputs = [index2source[i] for i in inputs.data.tolist()[0]]
    pred = [index2target[i] for i in pred.data.tolist()]

    print('Source : ', ' '.join([i for i in inputs if i not in ['</s>']]))
    print('Truth : ', ' '.join([i for i in truth.data.tolist()[0] if i not in [2, 3]]))
    print('Prediction : ', ' '.join([i for i in pred if i not in ['</s>']]))

    if USE_CUDA:
        atten = atten.cpu()

    attention_visualize.visualize(inputs, pred, atten.data)

if __name__ == "__main__":
    main()

