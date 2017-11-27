# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import USE_CUDA
from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 n_layers=1,
                 bidirec=False):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)

        if bidirec is True:
            self.n_direction = 2
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                              batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                              batch_first=True)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)

    def forward(self, inputs, input_lengths):
        hidden = self.init_hidden(inputs)

        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                                         batch_first=True)

        if self.n_layers >= 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]

        # Reference
        # unsqueeze
        #    http://pytorch.org/docs/master/torch.html#torch.unsqueeze
        return outputs, torch.cat(hidden, 1).unsqueeze(1)

