# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from utils import USE_CUDA
import torch.optim as optim
import random
import torch
import numpy as np
from utils import prepare_word
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
from utils import LongTensor
from utils import ByteTensor
from utils import pad_to_batch
from utils import get_batch
from utils import show_sentence


class Trainer(object):

    def __init__(self,
                 embedding_size: int=50,
                 batch_size: int=256,
                 epoch: int=50,
                 model: object=None,
                 hidden_size: int=512,
                 decoder_learning_rate: float=5.0,
                 lr: float=0.0001,
                 rescheduled: bool=False,
                 fine_tune_model: str=''
                 ):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch = epoch
        if USE_CUDA is True and model is not None:
            model = model.cuda()
        self.model = model
        if model is not None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        self.hidden_size = hidden_size
        self.decoder_learning_rate = decoder_learning_rate
        self.lr = lr
        self.rescheduled = rescheduled
        self.fine_tune_model = fine_tune_model

    def train_method(self, train_data: list):
        losses = []

        for epoch in range(self.epoch):
            for i, batch in enumerate(get_batch(batch_size=self.batch_size,
                                                train_data=train_data)):
                inputs, targets, coocs, weights = zip(*batch)

                inputs = torch.cat(inputs)
                targets = torch.cat(targets)
                coocs = torch.cat(coocs)
                weights = torch.cat(weights)
                self.model.zero_grad()

                loss = self.model(inputs, targets, coocs, weights)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.data.tolist()[0])
            if epoch % 10 == 0:
                print('Epoch : %d, mean loss : %.02f' % (epoch, np.mean(losses)))
                self.__save_model_info(inputs, epoch, losses)
                torch.save(self.model, './../models/glove_model_{0}.pth'.format(epoch))
                losses = []
        self.writer.add_graph(self.model, loss)
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def __save_model_info(self, inputs, epoch, losses):
        x = vutils.make_grid(inputs, normalize=True, scale_each=True)
        self.writer.add_image('Image', x, epoch)
        self.writer.add_scalar('data/scalar1', np.mean(losses), epoch)

    def train_attention(self,
                        train_data: list=[],
                        source2index: list=[],
                        target2index: list=[],
                        index2source: list=[],
                        index2target: list=[],
                        encoder_model: object=None,
                        decoder_model: object=None):

        encoder_model.init_weight()
        decoder_model.init_weight()

        encoder_model, decoder_model = self.__fine_tune_weight(encoder_model=encoder_model,
                                                               decoder_model=decoder_model)

        if USE_CUDA:
            encoder_model = encoder_model.cuda()
            decoder_model = decoder_model.cuda()

        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=self.lr)
        decoder_optimizer = optim.Adam(decoder_model.parameters(),
                                       lr=self.lr * self.decoder_learning_rate)
        for epoch in range(self.epoch):
            losses = []
            for i, batch in enumerate(get_batch(self.batch_size, train_data)):
                inputs, targets, input_lengths, target_lengths = \
                    pad_to_batch(batch, source2index, target2index)
                input_mask = torch.cat([Variable(ByteTensor(
                    tuple(map(lambda s: s == 0, t.data))))
                    for t in inputs]).view(inputs.size(0), -1)
                start_decode = Variable(LongTensor([[target2index['<s>']] * targets.size(0)])).transpose(0, 1)
                encoder_model.zero_grad()
                decoder_model.zero_grad()
                output, hidden_c = encoder_model(inputs, input_lengths)
                preds = decoder_model(start_decode, hidden_c, targets.size(1),
                                      output, input_mask, True)
                loss = loss_function(preds, targets.view(-1))
                losses.append(loss.data.tolist()[0])
                loss.backward()
                torch.nn.utils.clip_grad_norm(encoder_model.parameters(), 50.0)
                torch.nn.utils.clip_grad_norm(decoder_model.parameters(), 50.0)
                encoder_optimizer.step()
                decoder_optimizer.step()

                if i % 200 == 0:
                    test = random.choice(train_data)
                    inputs = test[0]

                    output_c, hidden = encoder_model(inputs, [inputs.size(1)])
                    show_preds, _ = decoder_model.decode(hidden, output_c, target2index, index2target)
                    show_preds = decoder_model(start_decode, hidden_c, targets.size(1),
                                          output, input_mask, True)
                    outputs = torch.max(show_preds, dim=1)[1].view(len(inputs), -1)
                    show_sentence(targets, inputs, outputs.data.tolist(), index2source, index2target)
                    print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" %(epoch,
                                                                    self.epoch,
                                                                    i,
                                                                    len(train_data) // self.batch_size,
                                                                    np.mean(losses)))
                    self.__save_model_info(inputs, epoch, losses)
                    torch.save(encoder_model, './../models/encoder_model_{0}.pth'.format(epoch))
                    torch.save(decoder_model, './../models/decoder_model_{0}.pth'.format(epoch))
                    losses=[]
                if self.rescheduled is False and epoch == self.epoch // 2:
                    self.lr = self.lr * 0.01
                    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=self.lr)
                    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=self.lr * self.decoder_learning_rate)
                    self.rescheduled = True
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def train_qrnn(self,
                   train_data: list=[],
                   source2index: list=[],
                   target2index: list=[],
                   index2source: list=[],
                   index2target: list=[],
                   qrnn_model: object=None):

        # qrnn_model.encoder, qrnn_model.decoder = self.__fine_tune_weight(
        #     encoder_model=qrnn_model.encoder,
        #     decoder_model=qrnn_model.decoder)
        if USE_CUDA:
            qrnn_model = qrnn_model.cuda()
            encoder_model = qrnn_model.encoder.cuda()
            decoder_model = qrnn_model.decoder.cuda()
            # proj_linear_model = qrnn_model.proj_linear.cuda()

        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        # qrnn_optimizer = optim.Adam(qrnn_model.parameters(), lr=self.lr)
        encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=self.lr)
        decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=self.lr)
        # proj_linear_optimizer = optim.Adam(proj_linear_model.parameters(), lr=self.lr)
        for epoch in range(self.epoch):
            losses = []
            for i, batch in enumerate(get_batch(self.batch_size, train_data)):
                inputs, targets, input_lengths, target_lengths = \
                    pad_to_batch(batch, source2index, target2index)
                qrnn_model.zero_grad()
                start_decode = Variable(LongTensor([[target2index['<s>']] * targets.size(1)]))
                preds = qrnn_model(inputs, input_lengths, start_decode)
                loss = loss_function(preds, targets.view(-1))
                losses.append(loss.data.tolist()[0])
                loss.backward()
                torch.nn.utils.clip_grad_norm(qrnn_model.parameters(), 50.0)
                # qrnn_optimizer.step()
                encoder_optimizer.step()
                decoder_optimizer.step()
                # proj_linear_optimizer.step()

                if i % 200 == 0:
                    test = random.choice(train_data)
                    show_inputs = test[0]
                    show_targets = test[1]
                    show_preds = qrnn_model(inputs, [inputs.size(1)], start_decode)
                    outputs = torch.max(show_preds, dim=1)[1].view(len(inputs), -1)
                    show_sentence(show_targets, show_inputs, outputs.data.tolist(), index2source, index2target)
                    print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" %(epoch,
                                                                    self.epoch,
                                                                    i,
                                                                    len(train_data) // self.batch_size,
                                                                    np.mean(losses)))
                    self.__save_model_info(inputs, epoch, losses)
                    torch.save(qrnn_model.encoder, './../models/test_qrnn_encoder_model_{0}.pth'.format(epoch))
                    torch.save(qrnn_model.decoder, './../models/test_qrnn_decoder_model_{0}.pth'.format(epoch))
                    torch.save(qrnn_model.proj_linear, './../models/test_qrnn_proj_linear_model_{0}.pth'.format(epoch))
                    losses=[]
                if self.rescheduled is False and epoch == self.epoch // 2:
                    self.lr = self.lr * 0.01
                    # qrnn_optimizer = optim.Adam(qrnn_model.parameters(), lr=self.lr)
                    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=self.lr)
                    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=self.lr * self.decoder_learning_rate)
                    # proj_linear_optimizer = optim.Adam(proj_linear_model.parameters(), lr=self.lr)
                    self.rescheduled = True
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def word_similarity(self,
                        target: list,
                        vocab: list,
                        word2index: dict,
                        top_rank: int=10
                        ):
        if USE_CUDA is True:
            target_V = self.model.prediction(prepare_word(target, word2index))
        else:
            target_V = self.model.prediction(prepare_word(target, word2index))
        similarities = []
        for i in range(len(vocab)):
            if vocab[i] == target:
                continue

            if USE_CUDA:
                vector = self.model.prediction(prepare_word(list(vocab)[i],
                                                            word2index))
            else:
                vector = self.model.prediction(prepare_word(list(vocab)[i],
                                                            word2index))
            consine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
            similarities.append([vocab[i], consine_sim])
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_rank]

    def __fine_tune_weight(self,
                         encoder_model: object=None,
                         decoder_model: object=None):
        fine_tune_model = torch.load(self.fine_tune_model)
        for name, module in fine_tune_model._modules.items():
            if name == 'embedding_v':
                w_v = list(module.parameters())
            if name == 'embedding_u':
                w_u = list(module.parameters())
        w = w_v + w_u
        encoder_model.embedding.weight = w[0]
        decoder_model.embedding.weight = w[0]
        del fine_tune_model
        del w

        return encoder_model, decoder_model
