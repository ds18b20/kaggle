from common.datasets import MNIST
from common.datasets import MNISTCSV


if __name__ == '__main__':
    mnist = MNIST('data\\MNIST')
    mnistcsv = MNIST('data\\MNIST')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    print(train_x.dtype, train_x.shape)
    print(train_y.dtype, train_y.shape)
    print(train_x[0][0][10])
    print(train_y[0])
    train_x, train_y, test_x, test_y = mnistcsv.load(normalize=True, image_flat=False, label_one_hot=False)
    print(train_x.dtype, train_x.shape)
    print(train_y.dtype, train_y.shape)
    print(train_x[0][0][10])
    print(train_y[0])