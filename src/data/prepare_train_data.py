# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from utils import prepare_word
from utils import FloatTensor
import torch
from torch.autograd import Variable
from typing import List


class PrepareTrainData(object):

    def prepare_train_data_method(self,
                                  window_data: List=[],
                                  word2index: dict={},
                                  weighting_dic: dict={},
                                  X_ik: dict={}
                                  ):
        u_p = []
        v_p = []
        co_p = []
        weight_p = []
        # Reference
        # view
        #    http://pytorch.org/docs/master/tensors.html#torch.Tensor.view
        for pair in window_data:
            u_p.append(prepare_word(pair[0], word2index).view(1, -1))
            v_p.append(prepare_word(pair[1], word2index).view(1, -1))
            try:
                cooc = X_ik[pair]
            except:
                cooc = 1

            co_p.append(torch.log(Variable(FloatTensor([cooc]))).view(1, -1))
            weight_p.append(Variable(
                FloatTensor([weighting_dic[pair]])).view(1, -1))

        train_data = list(zip(u_p, v_p, co_p, weight_p))
        return train_data
