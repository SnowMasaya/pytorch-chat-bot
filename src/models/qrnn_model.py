# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch.nn as nn
from models.encoder_qrnn import Encoder
from models.decoder_qrnn import Decoder
import torch.nn.functional as F
from utils import USE_CUDA
from utils import LongTensor
from torch.autograd import Variable
import torch


class QRNNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, hidden_size,
                 embedding_size, source_vocab_size, target_vocab_size,
                 use_attention=False,
                 zoneout=0.5, training=True, dropout=0.5,
                 ):
        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               embedding_size, source_vocab_size, use_attention=use_attention,
                               zoneout=zoneout, training=training, dropout=dropout)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               embedding_size, target_vocab_size,
                               zoneout=zoneout, training=training, dropout=dropout)
        self.proj_linear = nn.Linear(hidden_size, target_vocab_size)

    def encode(self, inputs, input_len):
        input_len = torch.LongTensor(input_len)
        if USE_CUDA:
            input_len = Variable(input_len.cuda())
        else:
            input_len = Variable(input_len)
        return self.encoder(inputs, input_len)

    def decode(self, inputs, init_states, memories):
        cell_states, hidden_size = self.decoder(inputs, init_states, memories)

        h_last = hidden_size[-1]

        return cell_states, self.proj_linear(h_last.view(-1, h_last.size(2)))

    def forward(self, encoder_inputs, encoder_len, decoder_inputs):
        init_states, memories = self.encode(encoder_inputs, encoder_len)

        _, logits = self.decode(decoder_inputs, init_states, memories)

        return logits

    def Attention(self, hidden, encoder_outputs):
        print(hidden[0])
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

    def decode_result(self, decoder_inputs, init_states, memories, target2index, index2target, max_length=50):
        start_decode = Variable(LongTensor([[target2index['<s>']] * 1])).transpose(0, 1)

        decodes = []
        embedded = start_decode
        embedd_list = []
        embedd_list.append(target2index['<s>'])

        # while decoded.data.tolist()[0] != target2index['</s>'] and max_length > len(decodes):
        for t in range(max_length):
            _, hidden = self.decode(embedded, init_states, memories)
            softmaxed = F.log_softmax(hidden)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedd_list.append(decoded.data.tolist()[0])
            embedded = Variable(LongTensor([embedd_list * 1]))
            if index2target[decoded.data.tolist()[0]] == '</s>' or (t != 0 and index2target[decoded.data.tolist()[0]] == '<s>'):
                break
            # context, alpha = self.Attention(hidden, decoder_inputs)
            # attentions.append(alpha.squeeze(1))
        print(embedded.size())

        return torch.cat(decodes).max(1)[1]
        # return torch.cat(decodes).max(1)[1], torch.cat(attentions)
