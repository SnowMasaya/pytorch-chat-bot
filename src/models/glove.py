# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch.nn as nn
import torch


class Glove(nn.Module):

    def __init__(self,
                 vocab_size,
                 projection_dim: int=50):
        super(Glove, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

        initrange = (2.0 / (vocab_size + projection_dim)) ** 0.5
        self.embedding_v.weight.data.uniform_(-initrange, initrange)
        self.embedding_u.weight.data.uniform_(-initrange, initrange)
        self.v_bias.weight.data.uniform_(-initrange, initrange)
        self.u_bias.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_words, target_words, coocs, weights):
        center_embeds = self.embedding_v(center_words)
        target_embeds = self.embedding_u(target_words)

        # Reference(squeeze)
        #    http://pytorch.org/docs/master/torch.html#torch.squeeze
        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.v_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # noqa

        loss = weights * torch.pow(inner_product + center_bias + target_bias - coocs, 2)  # noqa

        return torch.sum(loss)

    def prediction(self, inputs):
        v_embeds = self.embedding_v(inputs)
        u_embeds = self.embedding_u(inputs)

        return v_embeds + u_embeds
