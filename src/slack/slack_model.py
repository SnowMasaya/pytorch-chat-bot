# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from slackclient import SlackClient
import yaml
from collections import namedtuple
import os
import sys
from data.data_loader_attention import DataLoaderAttention
from models.qrnn_model import QRNNModel
from models.qrnn_layer import QRNNLayer
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class SlackModel():

    def __init__(self):
        """
        setting paramater Slack model
        :return:
        """
        self.Slack = namedtuple("Slack", ["api_token", "channel", "user", "image", "mecab"])
        Model = namedtuple("Model", ["train_data_name",
                                     "encoder_model_name",
                                     "decoder_model_name",
                                     "proj_linear_model_name"])
        self.config_file = "slack/conf/enviroment_slack.yml"
        self.config_model_file = "slack/conf/enviroment_model.yml"
        self.slack_channel = ""
        self.chan = ""
        self.user = ""
        self.mecab_dict = ""
        self.parameter_dict = {}
        with open(self.config_model_file, encoding="utf-8") as cf:
            e = yaml.load(cf)
            model = Model(e["model"]["train_data_name"],
                          e["model"]["encoder_model_name"],
                          e["model"]["decoder_model_name"],
                          e["model"]["proj_linear_model_name"],
                          )
            train_data_name = model.train_data_name
            encoder_model_name = model.encoder_model_name
            decoder_model_name = model.decoder_model_name
            proj_linear_model_name = model.proj_linear_model_name
        data_loader_attention = DataLoaderAttention(file_name=train_data_name)
        source2index, index2source, target2index, index2target, train_data = \
            data_loader_attention.load_data()
        self.source2index = source2index
        self.target2index = target2index
        self.index2source = index2source
        self.index2target = index2target

        HIDDEN_SIZE = 512
        NUM_LAYERS = 2
        KERNEL_SIZE = 2
        EMBEDDING_SIZE = 50
        SOURCE_VOCAB_SIZE = len(source2index)
        TARGET_VOCAB_SIZE = len(target2index)

        self.qrnn = QRNNModel(QRNNLayer, NUM_LAYERS, KERNEL_SIZE, HIDDEN_SIZE,
                              EMBEDDING_SIZE, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE)

        self.qrnn.encoder = torch.load(encoder_model_name, map_location=lambda storage, loc: storage)
        self.qrnn.decoder = torch.load(decoder_model_name, map_location=lambda storage, loc: storage)
        self.qrnn.proj_linear = torch.load(proj_linear_model_name, map_location=lambda storage, loc: storage)

    def read_config(self):
        """
        read config file for slack
        """
        with open(self.config_file, encoding="utf-8") as cf:
           e = yaml.load(cf)
           slack = self.Slack(e["slack"]["api_token"], e["slack"]["channel"],
                              e["slack"]["user"], e["slack"]["image"],
                              e["slack"]["mecab"])
           self.slack_channel = SlackClient(slack.api_token)
           self.chan = slack.channel
           self.user = slack.user
           self.image = slack.image
           self.mecab_dict = slack.mecab