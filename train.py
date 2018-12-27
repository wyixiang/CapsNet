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


def read_file():
    data = pd.read_csv('data/train.csv')
    #data['content'] = data.content.map(lambda x: ''.join(x.strip().split()))
    data['content'] = data.content.map(lambda x: ''.join(x.strip().split()))
    if remove_stop_words:
        with open('data/stop_words.txt') as f:
            stop_words = set([l.strip() for l in f])
            #print(stop_words)
        data['content'] = data.content.map(
            lambda x: ''.join([e for e in x if e not in stop_words]))

    # 把主题和情感拼接起来，一共10*3类
    #print(data.content[0:20])
    subj_to_id = {'动力':0, '价格':1, '内饰':2, '配置':3, '安全性':4, '外观':5, '操控':6, '油耗':7, '空间':8, '舒适性':9}
    sent_to_id = {-1:0, 0:10, 1:20}
    data['subject'] = data['subject'].apply(lambda x: subj_to_id.get(x))
    data['sentiment_value'] = data['sentiment_value'].apply(lambda x: sent_to_id.get(x))
    data['label'] = data['subject'] + data['sentiment_value']
    '''
    data['label'] = data['subject'] + data['sentiment_value'].astype(str)
    subj_lst = list(filter(lambda x: x is not np.nan, list(set(data.label))))
    subj_dic = {value: key for key, value in enumerate(subj_lst)}
    data['label'] = data['label'].apply(lambda x: subj_dic.get(x))
    #print(data[90:100])
    '''
    #print(data[90:100])
    return data


def process_data(data):
    data_tmp = data.groupby('content').agg({'label': lambda x: set(x)}).reset_index()
    #print(data_tmp[0:100])
    mlb = MultiLabelBinarizer()
    data_tmp['trans_label'] = mlb.fit_transform(data_tmp.label).tolist()
    Y = np.array(data_tmp.trans_label.tolist())

    bow = BOW(data_tmp.content.apply(jieba.lcut).tolist(), min_count=1, maxlen=100)

    word2vec = Word2Vec(data_tmp.content.apply(jieba.lcut).tolist(), size=300, min_count=1)
    word2vec.wv.save_word2vec_format('data/w2v.txt', binary=False)
    #word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/w2v.txt')

    vocab_size = len(bow.word2idx)
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for key, value in bow.word2idx.items():
        if key in word2vec.wv.vocab:
            embedding_matrix[value] = word2vec.wv.get_vector(key)
        else:
            embedding_matrix[value] = [0] * embedding_dim

    X = copy.deepcopy(bow.doc2num)
    Y = copy.deepcopy(Y)
    return X, Y, embedding_matrix, vocab_size


def my_f1_score(Y_true, Y_pred):
    Tp, Fp, Fn = 0.0, 0.0, 0.0
    for i in range(len(Y_pred)):
        y_true = Y_true[i]
        y_pred = Y_pred[i]

        Tp += np.sum(np.multiply(y_true, y_pred))
        Fp += np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
        Fn += np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
        # Tn += np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    P = Tp / (Tp + Fp)
    R = Tp / (Tp + Fn)
    return 2 * P * R / (P + R)


class BOW(object):
    def __init__(self, X, min_count=10, maxlen=100):
        self.X = X
        self.min_count = min_count
        self.maxlen = maxlen
        self.__word_count()
        self.__idx()
        self.__doc2num()

    def __word_count(self):
        wc = {}
        for text in tqdm(self.X, desc='   Word Count'):
            for x in text:
                if x in wc:
                    wc[x] += 1
                else:
                    wc[x] = 1
        self.word_count = {i: j for i, j in wc.items() if j >= self.min_count}

    def __idx(self):
        self.idx2word = {i + 1: j for i, j in enumerate(self.word_count)}
        self.word2idx = {j: i for i, j in self.idx2word.items()}

    def __doc2num(self):
        doc2num = []
        for text in tqdm(self.X, desc='Doc To Number'):
            s = [self.word2idx.get(i, 0) for i in text[:self.maxlen]]
            doc2num.append(s + [0] * (self.maxlen - len(s)))
        self.doc2num = np.asarray(doc2num)


def train_capnet(EPOCH, train_loader, test_content_tensor, Y_test, max_test_f1 = 0.5):
    #capnet = Capsule_Main(embedding_matrix, vocab_size)
    capnet = TextCNN(embedding_matrix, vocab_size)
    loss_func = nn.BCELoss()
    if USE_CUDA:
        capnet = capnet.cuda()
        loss_func.cuda()
    optimizer = Adam(capnet.parameters(), lr=LR, weight_decay=weight_decay)

    it = 1
    flag = 1
    threshold = 0.2
    #max_test_f1 = 0
    f1 = [0] * 9
    capnet.train()
    for epoch in tqdm_notebook(range(EPOCH)):
        for batch_id, (data, target) in enumerate(train_loader):
            # print(len(target.numpy()))
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            output = capnet(data)
            loss = loss_func(output, target)
            # print(np.rint(output.cpu().data.numpy()))
            if it % 50 == 0:
                capnet.eval()
                Y_test_pred = capnet(test_content_tensor.cuda()).cpu().data.numpy()
                for threshold_ in np.arange(0.1, 1, 0.1):
                    YY = copy.deepcopy(Y_test_pred)
                    YY[YY >= threshold_] = 1
                    YY[YY < threshold_] = 0
                    c = f1_score(Y_test, YY, average='micro')
                    i = int(10 * threshold_ - 1)
                    if f1[i] < c:
                        f1[i] = c
                Y_test_pred[Y_test_pred >= threshold] = 1
                Y_test_pred[Y_test_pred < threshold] = 0
                train_f1 = f1_score(np.rint(target.cpu().data.numpy()), np.rint(output.cpu().data.numpy()),
                                    average='micro')
                test_f1 = f1_score(Y_test, Y_test_pred, average='micro')
                print('\ntraining loss: ', loss.cpu().data.numpy().tolist())
                print('training f1: ', train_f1)
                print('    test f1: ', test_f1)
                if test_f1 > max_test_f1:
                    max_test_f1 = test_f1
                    torch.save(capnet, 'model_saved/capnet.pkl')
                capnet.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            it += 1

    print('max test f1: ', max_test_f1)
    print(f1)


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
    '''
    data = read_file()
    X, Y, embedding_matrix, vocab_size = process_data(data)
    print(vocab_size)
    torch.save(X, 'data/x.pth')
    torch.save(Y, 'data/y.pth')
    torch.save(embedding_matrix, 'data/embedding_matrix.pth')
    '''
    X = torch.load('data/x.pth')
    Y = torch.load('data/y.pth')
    embedding_matrix = torch.load('data/embedding_matrix.pth')
    vocab_size = 18691


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3)

    test_content_tensor = torch.from_numpy(np.array(X_test)).long()
    label_tensor = torch.from_numpy(np.array(Y_train)).float()
    content_tensor = torch.from_numpy(np.array(X_train)).long()
    torch_dataset = Data.TensorDataset(content_tensor, label_tensor)
    train_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=True,               # random shuffle for training
            num_workers=8,              # subprocesses for loading data
        )

    capnet = torch.load('model_saved/capnet.pkl').eval()
    Y_test_pred = capnet(test_content_tensor.cuda()).cpu().data.numpy()
    threshold = 0.2
    Y_test_pred[Y_test_pred >= threshold] = 1
    Y_test_pred[Y_test_pred < threshold] = 0
    test_f1 = f1_score(Y_test, Y_test_pred, average='micro')
    print('test f1: ', test_f1)


    train_capnet(EPOCH, train_loader, test_content_tensor, Y_test, test_f1)

    capnet = torch.load('model_saved/capnet.pkl').eval()
    Y_test_pred = capnet(test_content_tensor.cuda()).cpu().data.numpy()
    threshold = 0.2
    Y_test_pred[Y_test_pred >= threshold] = 1
    Y_test_pred[Y_test_pred < threshold] = 0
    test_f1 = f1_score(Y_test, Y_test_pred, average='micro')
    print('test f1: ', test_f1)