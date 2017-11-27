# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from utils import prepare_word


class Classifier(object):
    def __init__(self, model_name: str='', K: int=5,
                 output_classifier_name: str='glove_classify_model.pkl'):
        self.model = torch.load(model_name)
        self.classifier = MiniBatchKMeans(n_clusters=K, random_state=0)
        self.output_classifier_name = output_classifier_name
        self.other_classes = 9999

    def classify(self):
        weight = self.model.embedding_u.weight + self.model.embedding_v.weight
        classify_weight = self.__transfer_vector(weight)
        self.classifier.fit(classify_weight)
        joblib.dump(self.classifier, self.output_classifier_name)

    def classify_predict(self, word: str, classify_model_name: str, word2index: list):
        if word not in word2index:
            return self.other_classes
        vector = self.model.prediction(prepare_word(word,
                                                    word2index))
        vector = self.__transfer_vector(vector)
        classifier = joblib.load(classify_model_name)
        classes = classifier.predict(vector)
        return classes[0]

    def __transfer_vector(self, vector):
        vector = vector.cpu()
        vector = vector.data.numpy()
        return vector

