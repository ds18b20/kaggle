# coding: utf-8
# Reference: https://github.com/oreilly-japan/deep-learning-from-scratch
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import platform
from six.moves import cPickle as pickle
import sys


def one_hot(array, class_num=10):
    """
    (vec_count, )-->(vec_count, length)
    """
    array_shape = array.shape
    array_size = array.size

    vec = array.reshape(array_size, )
    ret = np.zeros((array_size, class_num))
    ret[range(array_size), vec] = 1

    return ret.reshape(*array_shape, class_num)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        # print('fxh1:', fxh1)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        # print('fxh2:', fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def get_one_batch(image_set, label_set, batch_size=100):
    """
    get batch data
    """
    set_size = len(image_set)
    index = np.random.choice(set_size, batch_size)
    return image_set[index], label_set[index]


def label2name(index_array, label_array):
    try:
        return label_array[index_array]
    except TypeError:
        print("Please check index type.")


def load_pickle(f, encoding='latin1'):
    version = platform.python_version_tuple()
    logging.info('Python version={}'.format(version[0]))
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding=encoding)
    raise ValueError("invalid python version: {}".format(version))


def show_accuracy_loss(train_acc, test_acc, train_loss, test_loss):
    n = 2
    _, figs = plt.subplots(1, n)
    # fig[0]: train accuracy & test accuracy
    figs[0].plot(train_acc, label='train accuracy')
    figs[0].plot(test_acc, label='test accuracy')
    figs[0].legend()
    # fig[1]: train loss & test loss
    figs[1].plot(train_loss, label='train loss')
    figs[1].plot(test_loss, label='test loss')
    figs[1].legend()
    plt.show()


def show_imgs(images, titles):
    logging.info('M@{}, F@{}, show images: {}'.format(__name__, sys._getframe().f_code.co_name, titles))

    if images.ndim == 4 and images.shape[3] == 1:
        images_show = np.squeeze(images, axis=(3,))
    else:
        images_show = images

    n = images_show.shape[0]
    # _, figs = plt.subplots(1, n, figsize=(15, 15))
    _, figs = plt.subplots(1, n)
    for i in range(n):
        figs[i].imshow(images_show[i])
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
        figs[i].axes.set_title(titles[i])
    plt.show()


def show_img(window_title="log"):
    """
    coroutine by generator
    Show images in new window
    """
    while True:
        image, label = (yield)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, image)
        cv2.waitKey(0)  # pause here
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def show_filter(filters, ncols=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    nrows = int(np.ceil(FN / ncols))
    if FN < ncols:
        ncols = FN

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for index in range(FN):
        # https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html?highlight=add_subplot#matplotlib.figure.Figure.add_subplot
        ax = fig.add_subplot(nrows, ncols, index+1, xticks=[], yticks=[])
        ax.imshow(filters[index, 0], cmap=cm.gray_r, interpolation='nearest')
    plt.show()


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる
    Use covolution to smooth input data
    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # logging.info("start im2col...")
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # logging.info("col shape: {}".format(col.shape))

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == '__main__':
    step_num = 2
    batch_size = 3
    class_num = 5
    input_array = np.random.choice(class_num, size=(step_num, batch_size))
    output_array = one_hot(input_array, class_num=class_num)
    print('input_array:\n', input_array)
    print('output_array:\n', output_array)