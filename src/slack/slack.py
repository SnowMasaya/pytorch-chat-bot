# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import time
from slack_model import SlackModel
import sys
import os
import MeCab
from utils import prepare_sequence
import torch
from torch.autograd import Variable
from utils import LongTensor
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class SlackApp():
    """
    Slack Call app
    You preapre the chainer model, You execute the bellow command, you can play the dialogue app
    Example
        python app.py
    """

    def __init__(self, data_model):
        """
        Iniital Setting
        :param data_model: Setting Slack Model. Slack Model has the a lot of paramater
        """
        self.slack_channel = data_model.slack_channel
        self.data = ""
        self.chan = data_model.chan
        self.usr = data_model.user
        self.icon_url = data_model.image
        self.mecab_dict = data_model.mecab_dict
        self.qrnn = data_model.qrnn
        self.data_model = data_model
        self.Mecab = MeCab.Tagger("-Owakati -d %s" % self.mecab_dict)

    def call_method(self):
        """
        Slack api call
        1: read sentence
        2: model return the sentence
        """
        if self.slack_channel.rtm_connect():
            while True:
                self.data = self.slack_channel.rtm_read()
                self.__judge_print()
                time.sleep(1)
        else:
            print("connection Fail")

    def __judge_print(self):
        """
        judge slack call for chainer
        Example:
            chainer:{your sentence}
                chainer return the sentence
            chainer_train:{your sentence}
                start train
        """
        if len(self.data) >= 1 and "text" in self.data[0]:
            print(self.data[0]["text"])
            if "pytorch:" in self.data[0]["text"]:
                # input sentence
                src_batch = self.__input_sentence()
                # predict
                hyp_batch = self.__predict_sentence(src_batch)
                # show predict word
                print(self.slack_channel.api_call("chat.postMessage", user=self.usr, channel=self.chan, text=hyp_batch, icon_url=self.icon_url))

    def __input_sentence(self):
        """
        return sentence for chainer predict
        """
        text = self.__mecab_method(self.data[0]["text"].replace("pytorch:", ""))
        return text

    def __predict_sentence(self, src_batch):
        """
        predict sentence
        :param src_batch: get the source sentence
        :return:
        """
        hyp_batch = ''

        inputs = prepare_sequence(['<s>'] + src_batch + ['</s>'], self.data_model.source2index).view(1, -1)
        start_decode = Variable(LongTensor([[self.data_model.target2index['<s>']] * inputs.size(1)]))
        show_preds = self.qrnn(inputs, [inputs.size(1)], start_decode)
        outputs = torch.max(show_preds, dim=1)[1].view(len(inputs), -1)
        for pred in outputs.data.tolist():
            for each_pred in pred:
                hyp_batch += self.data_model.index2target[each_pred]
        hyp_batch = hyp_batch.replace('<s>', '')
        hyp_batch = hyp_batch.replace('</s>', '')
        return hyp_batch

    def __mecab_method(self, text):
        """
        Call the mecab method
        :param text: user input text
        :return:
        """
        mecab_text = self.Mecab.parse(text)
        return mecab_text.split(" ")

if __name__ == '__main__':
    data_model = SlackModel()
    data_model.read_config()
    slack = SlackApp(data_model)
    slack.call_method()
