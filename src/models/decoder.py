# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch.nn as nn
from torch.autograd import Variable
from utils import USE_CUDA
from utils import LongTensor
import torch.nn.functional as F
import torch


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 n_layers=1,
                 dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layess = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size,
                          n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, input_size)
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layess, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attention.weight = nn.init.xavier_uniform(self.attention.weight)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        hidden = hidden[0].unsqueeze(2)

        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        # reference
        # contiguous
        #    http://pytorch.org/docs/master/tensors.html#torch.Tensor.contiguous
        energies = self.attention(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        energies = energies.view(batch_size, max_len, -1)
        attention_energies = energies.bmm(hidden).squeeze(2)

        alpha = F.softmax(attention_energies)
        alpha = alpha.unsqueeze(1)
        context = alpha.bmm(encoder_outputs)

        return context, alpha

    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        if is_training is True:
            embedded = self.dropout(embedded)

        decode = []

        for i in range(max_length):
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.softmax(score)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            if is_training is True:
                embedded = self.dropout(embedded)

            context, alpha = self.Attention(hidden, encoder_outputs, encoder_maskings)

        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)

    def decode(self, context, encoder_outputs, target2index, index2target, max_lengh=50):
        start_decode = Variable(LongTensor([[target2index['<s>']] * 1])).transpose(0, 1)
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)

        decodes = []
        attentions = []
        decoded = embedded

        while decoded.data.tolist()[0] != target2index['</s>'] and max_lengh > len(attentions):
            _, hieedn = self.gru(torch.cat((embedded, context), 2), hidden)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            # print(index2target[decoded.data.tolist()[0]])
            embedded = self.embedding(decoded).unsqueeze(1)
            context, alpha = self.Attention(hidden, encoder_outputs, None)
            attentions.append(alpha.squeeze(1))

        return torch.cat(decodes).max(1)[1], torch.cat(attentions)