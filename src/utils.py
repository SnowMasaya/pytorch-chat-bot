# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from torch.autograd import Variable
import codecs
import torch
import shutil
import random
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

def pad_to_batch(batch, x_to_ix, y_to_ix):
    sorted_batch = sorted(batch, key=lambda b:b[0].size(1), reverse=True)
    x,y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i],
                                  Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
        if y[i].size(1) < max_y:
            y_p.append(torch.cat([y[i],
                                  Variable(LongTensor([y_to_ix['<PAD>']] * (max_y - y[i].size(1)))).view(1, -1)], 1))
        else:
            y_p.append(y[i])

    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s == 0, t.data)).count(False) for t in input_var]
    target_len = [list(map(lambda s: s == 0, t.data)).count(False) for t in target_var]

    return input_var, target_var, input_len, target_len

def get_batch(
              batch_size:int,
              train_data: list):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def show_sentence(targets, inputs, preds, index2source, index2target):
    for target, input, pred in zip(targets, inputs, preds):
        source =''
        replay =''
        pred_replay =''
        for each_input, each_target, each_pred in zip(input.data, target.data, pred):
            source += index2source[each_input]
            replay += index2target[each_target]
            pred_replay += index2target[each_pred]
        print('------decode----------------')
        print('source: {0}'.format(source))
        print('replay: {0}'.format(replay))
        print('replay_pred: {0}'.format(pred_replay))
        print('--------------------------')
