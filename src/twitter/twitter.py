# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import time
from twitter_model import TwitterModel
import tweepy
import sys
import os
import MeCab
from utils import prepare_sequence
import torch
from torch.autograd import Variable
from utils import LongTensor
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class TwitterApp(tweepy.StreamListener):
    """
    Twitter Call app
    You preapre the chainer model, You execute the bellow command, you can play the dialogue app
    Example
        python app.py
    """

    def __init__(self, data_model, api):
        """
        Iniital Setting
        :param data_model: Setting Twitter Model. Twitter Model has the a lot of paramater
        """
        self.data = ""
        self.mecab_dict = data_model.mecab_dict
        self.qrnn = data_model.qrnn
        self.data_model = data_model
        self.Mecab = MeCab.Tagger("-Owakati -d %s" % self.mecab_dict)
        super().__init__(api)
        self.me = self.api.me()

    def on_data(self, data):
        try:
            tweet = json.loads(data)
            if self.data_model.twitter_name in tweet['text']:
                text = str("@%s %s " % (tweet['user']['screen_name'], self.__judge_print(tweet['text'])))
                print(text)
                self.api.update_status(status=text, in_reply_to_status_id=tweet['user']['id'])
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def __judge_print(self, text):
        """
        judge twitter call for chainer
        Example:
            chainer:{your sentence}
                chainer return the sentence
            chainer_train:{your sentence}
                start train
        """
        if len(text) >= 1:
            # input sentence
            src_batch = self.__input_sentence(text)
            # predict
            hyp_batch = self.__predict_sentence(src_batch)
            # show predict word
            return hyp_batch

    def __input_sentence(self, raw_text):
        """
        return sentence for chainer predict
        """
        text = self.__mecab_method(raw_text)
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
    data_model = TwitterModel()
    data_model.read_config()
    api = tweepy.API(data_model.auth)
    stream = tweepy.Stream(auth=api.auth, listener=TwitterApp(data_model, api))
    stream.userstream()
