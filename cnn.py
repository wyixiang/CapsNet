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
maxlen = 100
windows_size = [3,4,5,6]


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(TextCNN, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=maxlen,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=100 - h + 1))
            for h in windows_size
        ])


        self.fc1 = nn.Sequential(
            nn.Linear(len(windows_size)*maxlen, 400),
            nn.ReLU(),
            nn.Dropout(p=dropout_p, inplace=True)
        )

        self.fc = nn.Linear(400, num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed_x = self.embed_layer(x)
        embed_x = embed_x.permute(0, 2, 1)
        #x = self.cons(x)
        #x = self.conv2(x)

        # collapse
        out = [conv(embed_x) for conv in self.convs]
        x = torch.cat(out, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc(x)
        x = self.Sigmoid(x)

        return x