#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings
import torch.utils.data as Data
import torch.nn as nn
import torch
from torch.optim import Adam
import numpy as np
import pandas as pd
import jieba
import gensim
from gensim.models import Word2Vec, FastText
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import copy
import time

from capsule import *
from cnn import *


USE_CUDA = True
embedding_dim = 300
LR = 0.001
num_classes = 30
remove_stop_words = True
weight_decay = 1e-4

EPOCH = 200
BATCH_SIZE = 64


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.set_printoptions(precision=2, suppress=True, threshold=np.nan)

    X = torch.load('data/x.pth')
    Y = torch.load('data/y.pth')
    embedding_matrix = torch.load('data/embedding_matrix.pth')
    vocab_size = 18691


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3)

    test_content_tensor = torch.from_numpy(np.array(X_test)).long()
    label_tensor = torch.from_numpy(np.array(Y_train)).float()
    content_tensor = torch.from_numpy(np.array(X_train)).long()
    torch_dataset = Data.TensorDataset(content_tensor, label_tensor)

    capnet = torch.load('model_saved/capnet59.pkl').eval()
    Y_test_pred1 = capnet(test_content_tensor.cuda()).cpu().data.numpy()

    capnet = torch.load('model_saved/capnet58.pkl').eval()
    Y_test_pred2 = capnet(test_content_tensor.cuda()).cpu().data.numpy()

    cnn = torch.load('model_saved/cnn2fc1.pkl').eval()
    Y_test_pred3 = cnn(test_content_tensor.cuda()).cpu().data.numpy()

    cnn = torch.load('model_saved/cnn2fc2.pkl').eval()
    Y_test_pred4 = cnn(test_content_tensor.cuda()).cpu().data.numpy()

    for threshold in np.arange(0.1, 1, 0.02):
        Y_test_pred = Y_test_pred1 + Y_test_pred2 + Y_test_pred3 + Y_test_pred4
        Y_test_pred = Y_test_pred / 2

        Y_test_pred[Y_test_pred >= threshold] = 1
        Y_test_pred[Y_test_pred < threshold] = 0
        test_f1 = f1_score(Y_test, Y_test_pred, average='micro')
        print('%.2f test f1: %.5f'% (threshold, test_f1))