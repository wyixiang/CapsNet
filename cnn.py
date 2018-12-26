#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import time


use_pretrained_embedding = True
BATCH_SIZE = 128
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28
T_epsilon = 1e-7
num_classes = 30

word_embedding_dimension = 300


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(TextCNN, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.conv1 = nn.Sequential(
            nn.Conv1d(100, 400, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(118400, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p, inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p, inplace=True)
        )

        self.fc3 = nn.Linear(1024, num_classes)
        #self.log_softmax = nn.LogSoftmax()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.conv5(x)
        #x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        #x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.Sigmoid(x)

        return x