#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from common import layers
from common.util import numerical_gradient, get_one_batch
from common.optimizer import SGD, Adam
from common.datasets import HousePrices

import functools
import time


class MultiLayerRegression(object):
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # weights initialization
        self.__init_weight(weight_init_std)

        # generate layers
        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)],
                                                             self.params['b' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)],
                                                         self.params['b' + str(idx)])

        # self.last_layer = layers.SoftmaxCrossEntropy()
        self.last_layer = layers.MSE()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            print(type(weight_init_std), ':', weight_init_std)
            if isinstance(weight_init_std, str):
                if weight_init_std.lower() in ['sigmoid', 'xavier']:
                    scale = np.sqrt(1.0 / all_size_list[idx - 1])
                elif weight_init_std.lower() in ['relu', 'he']:
                    scale = np.sqrt(2.0 / all_size_list[idx - 1])
                else:
                    print('str ')

            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b'+str(idx)] = np.random.randn(all_size_list[idx])

    def predict(self, x_batch):
        tmp = x_batch.copy()  # .copy() is not necessary!
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
        return tmp

    def loss(self, x_batch, t_batch):
        y_batch = self.predict(x_batch)
        loss = self.last_layer.forward(y_batch, t_batch)
        return loss

    def gradient(self, x_batch, t_batch):
        # forward
        self.loss(x_batch, t_batch)
        # backward
        dout = 1
        dout = self.last_layer.backward(d_y=dout)
        layers = list(self.layers.values())  # self.layers is a dict not a list!!!
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        grad = {}
        for idx in range(1, len(self.hidden_size_list) + 2):
            grad['W'+str(idx)] = self.layers['Affine' + str(idx)].d_W
            grad['b'+str(idx)] = self.layers['Affine' + str(idx)].d_b
        return grad
    
    def numerical_gradient(self, x, t):
        self.x = x
        self.t = t
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads


if __name__ == '__main__':
    hp_data = HousePrices('./data')
    x, t, x_submission = hp_data.load()
    train_num = 1000
    train_x, train_y, test_x, test_y = x[:train_num, :], t[:train_num, :], x[train_num:, :], t[train_num:, :]
    
    max_iterations = 2000
    batch_size = 128
    # initialize network optimizer
    weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
    optimizer = SGD(lr=0.01)

    network = MultiLayerRegression(input_size=331, hidden_size_list=[100, 100, 100, 100],
                                   output_size=1, weight_init_std='relu', activation='relu')
    train_loss = []

    # 2:訓練の開始==========
    for i in range(max_iterations):
        x_batch, t_batch = get_one_batch(train_x, train_y, batch_size=batch_size)
            
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        loss = network.loss(x_batch, t_batch)
        train_loss.append(loss)
        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            # loss = network.loss(x_batch, t_batch)
            print('Sigmoid-Xavier' + ":" + str(loss))
    # x_pre = train_x[0:3]
    # t_pre = train_y[0:3]
    # y_pre = network.predict(x_pre)
    # print('label:', t_pre)
    # print('prediction:', y_pre)
    y_submission = 10**network.predict(x_submission)
    df = pd.DataFrame({'Id': hp_data.test_id, 'SalePrice': y_submission.flatten()})
    df.to_csv('prediction.csv', index=False)
