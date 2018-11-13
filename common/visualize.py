# coding: utf-8
# Reference: https://github.com/oreilly-japan/deep-learning-from-scratch
# import logging; logging.basicConfig(level=logging.INFO)
# import logging; logging.basicConfig(level=logging.DEBUG)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


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


if __name__ == '__main__':
    print('OK')
