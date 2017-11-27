# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


class AttentionVisualize(object):

    def __init__(self,
                 encoder_model_name: str='',
                 decoder_model_name: str=''
                 ):
        self.encoder_model = torch.load(encoder_model_name)
        self.decoder_model = torch.load(decoder_model_name)
        self.fig = plt.figure()

    def visualize(self, input_words, output_words, attentions):
        ax = self.fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='done')
        self.fig.colorbar(cax)

        ax.set_xticklabels([''] + input_words, rotation=90)
        ax.set_yticklabels([''] + output_words)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()

