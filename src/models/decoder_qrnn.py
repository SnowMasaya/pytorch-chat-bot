# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
from utils import USE_CUDA
import torch


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, embedding_size, target_vocab_size,
                 zoneout, training, dropout,
                 ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        layers = []
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                input_size = embedding_size
            else:
                input_size = hidden_size
            if layer_idx == n_layers - 1:
                use_attention = True
            else:
                use_attention = False
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, use_attention,
                                     zoneout, training, dropout))
        self.layers = nn.Sequential(*layers)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layess, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, inputs, init_states, memories):
        assert len(self.layers) == len(memories)

        h = self.embedding(inputs)

        cell_states, hidden_states = [], []

        for layer_idx, layer in enumerate(self.layers):
            if init_states is None:
                state = None
            else:
                state = init_states[layer_idx]
            memory = memories[layer_idx]

            c, h = layer(h, state, memory)
            cell_states.append(c)
            hidden_states.append(h)

        return cell_states, hidden_states

