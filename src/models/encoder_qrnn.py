# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, embedding_size, source_vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(source_vocab_size, embedding_size)
        layers = []
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                input_size = embedding_size
            else:
                input_size = hidden_size
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, False))
        self.layers = nn.Sequential(*layers)

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, inputs, input_len):
        h = self.embedding(inputs)

        cell_states, hidden_states = [], []
        for layer in self.layers:
            c, h = layer(h)
            time = Variable(
                torch.arange(0, h.size(1)).unsqueeze(-1).expand_as(h).long())
            if h.is_cuda:
                time = time.cuda()

            mask = (input_len.unsqueeze(-1).unsqueeze(-1) > time).float()
            h = h * mask

            c_last = c[range(len(inputs)), (input_len - 1).data, :]
            h_last = h[range(len(inputs)), (input_len - 1).data, :]
            cell_states.append(c_last)
            hidden_states.append((h_last, h))

        return cell_states, hidden_states
