# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from visualization.qrnn_visualize import QrnnVisualize
from data.data_loader_attention import DataLoaderAttention
from models.qrnn_model import QRNNModel
from models.qrnn_layer import QRNNLayer
import random
import argparse
import torch
from utils import show_sentence
from torch.autograd import Variable
from utils import LongTensor
torch.backends.cudnn.enabled = False


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-n", "--num_layers", metavar="num_layers",
                        type=int, default=2,
                        dest="num_layers", help="set the layer number")
    parser.add_argument("-k", "--kernel_size", metavar="kernel_size",
                        type=int, default=2,
                        dest="kernel_size", help="set the kernel_size")
    batch_size = 64
    args = parser.parse_args()
    test_data_loader_attention = DataLoaderAttention(file_name=args.train_data)
    source2index, index2source, target2index, index2target, train_data = \
        test_data_loader_attention.load_data()

    encoder_model_name = '../models/qrnn_encoder_model_285.pth'
    decoder_model_name = '../models/qrnn_decoder_model_285.pth'
    proj_linear_model_name = '../models/qrnn_proj_linear_model_285.pth'

    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    KERNEL_SIZE = args.kernel_size
    EMBEDDING_SIZE = args.embedding_size
    SOURCE_VOCAB_SIZE = len(source2index)
    TARGET_VOCAB_SIZE = len(target2index)
    ZONE_OUT = 0.0
    TRAINING = False
    DROPOUT = 0.0

    qrnn = QRNNModel(QRNNLayer, NUM_LAYERS, KERNEL_SIZE, HIDDEN_SIZE,
                     EMBEDDING_SIZE, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                     ZONE_OUT, TRAINING, DROPOUT
                     )

    qrnn.encoder = torch.load(encoder_model_name)
    qrnn.decoder = torch.load(decoder_model_name)
    qrnn.proj_linear = torch.load(proj_linear_model_name)

    test = random.choice(train_data)
    inputs = test[0]
    truth = test[1]
    print(inputs)
    print(truth)

    start_decode = Variable(LongTensor([[target2index['<s>']] * truth.size(1)]))
    show_preds = qrnn(inputs, [inputs.size(1)], start_decode)
    outputs = torch.max(show_preds, dim=1)[1].view(len(inputs), -1)
    show_sentence(truth, inputs, outputs.data.tolist(), index2source, index2target)

    # if USE_CUDA:
    #     atten = atten.cpu()

    # qrnn_visualize.visualize(inputs, pred, atten.data)

if __name__ == "__main__":
    main()

