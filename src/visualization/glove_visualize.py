# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from tensorboardX import SummaryWriter
from utils import prepare_word
from utils import USE_CUDA
import torch


class GloveVisualize(object):

    def __init__(self,
                 model_name: str=''
                 ):
        self.writer = SummaryWriter()
        self.model = torch.load(model_name)

    def visualize(self, vocab):
        for name, module in self.model._modules.items():
            if name == 'embedding_v':
                w_v = list(module.parameters())
            if name == 'embedding_u':
                w_u = list(module.parameters())
        w = w_v + w_u
        vector = w[0].data
        self.writer.add_embedding(vector, metadata=vocab)
        self.writer.close()
