#!/usr/bin/env python
# encoding: utf-8


import cPickle
import gzip

import numpy as np


def mnist_load():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return (training_data, validation_data, test_data)

def cost_function(x, y, w, m, lamb):
    rsp = x * w
    rsp = np.exp(rsp - rsp.max(1))
    normal = - np.multiply(y, (rsp * 1.0 / rsp.sum(1))).sum() / m
    return normal + lamb * w.sum() / 2.0

def learn(train_x, train_y, learning_rate, lamb, num_class, max_iter):
    m, n = train_x.shape
    x = train_x # m * (feature_number + 1)
    y = train_y
    w = np.mat(np.zeros((n, num_class))) # (feature_number + 1) * 2
    # w = np.mat(np.random.randn(n, num_class))
    for k in xrange(max_iter):
        w = SGD(x, y, w, learning_rate, lamb)
    return w

def SGD(x, y, w, learning_rate, lamb):
    m, n = x.shape
    for i in xrange(m):
        rsp = x[i, :] * w
        rsp = np.exp(rsp - rsp.max())
        sum_p = rsp.sum()
        h = rsp * 1.0 / sum_p
        g = - x[i, :].T * (y[i, :] - h) + lamb * w
        w = w - g * learning_rate
    return w

def BGD(x, y, w, learning_rate, lamb):
    m, n = x.shape
    rsp = x * w # m x num_class
    rsp = np.exp(rsp - rsp.max(1))
    sum_p = rsp.sum(1)
    h = rsp * 1.0 / sum_p
    g = - x.T * (y - h) / m + lamb * w
    w = w - g * learning_rate
    return w

def predict(test_x, test_y, w):
    m, n = test_x.shape
    rsp = test_x * w # m * 2
    pred = np.argmax(rsp, 1)
    acc = ((np.mat(pred).T == np.mat(test_y)) * 1.0).sum() / m
    return acc

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_load()

    train_x = np.mat(training_data[0])
    train_y = np.mat(training_data[1])
    # validation_x = np.mat(validation_data[0])
    # validation_y = np.mat(validation_data[1])
    # train_x = np.vstack((train_x, validation_x))
    bias = np.ones((train_x.shape[0], 1))
    train_x = np.hstack((train_x, bias))
    # train_y = np.hstack((train_y, validation_y))
    train_y = (train_y.T == np.mat(range(10))) * 1.0

    test_x = np.mat(test_data[0])
    bias = np.ones((test_x.shape[0], 1))
    test_x = np.hstack((test_x, bias))
    test_y = test_data[1]

    acc = []
    # BGD
    # learning_rate = 0.1
    # lamb = 0.01

    # SGD
    learning_rate = 0.005
    lamb = 0.005
    w = learn(train_x, train_y, learning_rate, lamb, 10, 1)
    ans = predict(test_x, test_y, w)
    print 'learning_rate=', learning_rate, '  lamb=', lamb, '  acc=', ans
    acc.append(ans)


