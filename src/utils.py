# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from torch.autograd import Variable
import codecs
import torch
import shutil
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def read_file(file_name: str):
    with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        read_data = f.read().split('\n')
    return read_data


def prepare_word(word: str, word2index: dict):
    return Variable(LongTensor([word2index[word]]) if word in word2index.keys()
                    else LongTensor(word2index['<UNK>']))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if w in to_index.keys() else to_index['<UNK>'], seq))
    return Variable(LongTensor(idxs))