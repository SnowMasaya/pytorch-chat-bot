# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from data.data_loader import DataLoader
from data.prepare_train_data import PrepareTrainData
from models.glove import Glove
from models.trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Training glove model")

    parser.add_argument("-c", "--train_data", metavar="train_data",
                        # type=str, default='../data/raw/jawiki_only_word_random_choose.txt',
                        type=str, default='../data/processed/twitter_vocab.txt',
                        dest="train_data", help="set the training data ")
    parser.add_argument("-e", "--embedding_size", metavar="embedding_size",
                        type=int, default=300,
                        dest="embedding_size", help="set the embedding size")
    args = parser.parse_args()

    data_loader = DataLoader()
    japanese_wiki_data = args.train_data
    word2index, index2word, window_data, X_ik, weightinhg_dict = \
        data_loader.load_data(file_name=japanese_wiki_data)  # noqa

    prepare_train_data = PrepareTrainData()
    train_data = \
        prepare_train_data.prepare_train_data_method(
            window_data=window_data,
            word2index=word2index,
            weighting_dic=weightinhg_dict,
            X_ik=X_ik)

    model = Glove(vocab_size=len(word2index), projection_dim=args.embedding_size)
    trainer = Trainer(model=model)

    trainer.train_method(train_data=train_data)

    word_similarity = trainer.word_similarity(
        target=data_loader.vocab[0],
        vocab=data_loader.vocab,
        word2index=word2index,
        top_rank=2
    )
    print(word_similarity)

if __name__ == "__main__":
    main()
