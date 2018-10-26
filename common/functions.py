#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


def mean_squared_error(y, label):
    # y/label can be multi dimensional array
    row_count = y.shape[0]
    ret = 0.5 * np.sum((y - label) ** 2) / row_count

    return ret  # sum((y-t)**2) / row_count


def softmax(array):
    tmp = array.copy()
    tmp -= tmp.max(axis=1, keepdims=True)  # max of array in axis_1(batch direction)
    exp = np.exp(tmp)  # exp(matrix)
    return exp / np.sum(exp, axis=1, keepdims=True)  # exp / sum of each row


def cross_entropy(y, label):
    # slice prediction result by label
    # TODO
    # y/label can be multi dimensional array ???
    assert y.shape[0] == label.shape[0]
    delta = 1e-6  # in case of log(0)
    
    row_count = y.shape[0]
    index_row = range(row_count)
    index_column = label.flatten()  # Error Fixed: label must be a one dimensional array
    picked = y[index_row, index_column] + delta  # choose prediction corresponding to label
    return np.sum(-np.log(picked)) / row_count  # sum(-t * ln(y)) / row_count


def accuracy(y_hat: np.array, y: np.array):
    tmp = y_hat.argmax(axis=1) == y  # type: np.ndarray
    return np.mean(tmp)


def check_is_ndarray(input_array):
    try:
        return isinstance(input_array, np.ndarray)  # or use: hasattr(input_array, shape)
    except TypeError:
        print("Input data type should be ndarray.")