# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.autograd import Variable


class QRNNLayer(nn.Module):
    """
    Reference
        https://github.com/JayParks/quasi-rnn
    """
    def __init__(self, input_size, hidden_size, kernel_size, use_attetion=False,
                 zoneout=0.5, training=True, dropout=0.5):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_attention = use_attetion

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=3*hidden_size,
                                kernel_size=kernel_size)

        self.conv_linear = nn.Linear(hidden_size, 3 * hidden_size)
        self.rnn_linear = nn.Linear(2*hidden_size, hidden_size)
        self.zoneout = zoneout
        self.training = training
        self.dropout = dropout

    def _conv_step(self, inputs, memory=None):
        inputs_ = inputs.transpose(1, 2)

        padded = FF.pad(inputs_.unsqueeze(2), (self.kernel_size-1, 0, 0, 0)).squeeze(2)

        gates = self.conv1d(padded).transpose(1, 2)
        if memory is not None:
            gates = gates + self.conv_linear(memory).unsqueeze(1)

        Z, F, O = gates.split(split_size=self.hidden_size, dim=2)
        return Z.tanh(), F.sigmoid(), O.sigmoid()

    def _rnn_step(self, z, f, o, c, attention_memory=None):
        if c is None:
            c_ = (1 - f) * z
        else:
            c_ = f * c + (1 - f) * z

        if not self.use_attention:
            return c_, (o * c_)

        alpha = FF.softmax(torch.bmm(c_, attention_memory.transpose(1, 2)).squeeze(1))
        context = torch.sum(alpha.unsqueeze(-1) * attention_memory, dim=1)
        h_ = self.rnn_linear(torch.cat([c_.squeeze(1), context], dim=1)).unsqueeze(1)

        return c_, (o * h_)

    def forward(self, inputs, state=None, memory=None):
        if state is None:
            c = None
        else:
            c = state.unsqueeze(1)

        if memory is None:
            (conv_memory, attention_memory) = (None, None)
        else:
            (conv_memory, attention_memory) = memory

        Z, F, O = self._conv_step(inputs, conv_memory)
        if self.training:
            mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout),
                            requires_grad=False)
            F = F * mask

        c_time, h_time = [], []
        # Reference
        # split
        #     http://pytorch.org/docs/master/torch.html?highlight=split#torch.split
        for time, (z, f, o) in enumerate(zip(Z.split(1, 1), F.split(1, 1), O.split(1, 1))):
            c, h = self._rnn_step(z, f, o, c, attention_memory)

            if self.dropout != 0 and self.training:
                c = torch.nn.functional.dropout(c, p=self.dropout,
                                                training=self.training,
                                                inplace=False)
            c_time.append(c)
            h_time.append(h)

        return torch.cat(c_time, dim=1), torch.cat(h_time, dim=1)