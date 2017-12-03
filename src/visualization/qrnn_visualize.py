# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from matplotlib.font_manager import FontProperties


class QrnnVisualize(object):

    def __init__(self,
                 qrnn_model_encoder_name: str='',
                 qrnn_model_decoder_name: str='',
                 qrnn_model_proj_linear_name: str='',
                 ):
        self.qrnn_model_encoder = torch.load(qrnn_model_encoder_name)
        self.qrnn_model_decoder = torch.load(qrnn_model_decoder_name)
        self.qrnn_model_proj_linear = torch.load(qrnn_model_proj_linear_name)
        self.fig = plt.figure()

    def visualize(self, input_words, output_words, attentions):
        fp = FontProperties(fname='visualization/conf/ipag.ttf', size=14)
        ax = self.fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        self.fig.colorbar(cax)

        ax.set_xticklabels([''] + input_words, rotation=90, fontproperties=fp)
        ax.set_yticklabels([''] + output_words, fontproperties=fp)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()

