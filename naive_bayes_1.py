#!/usr/bin/env python
# encoding: utf-8


import numpy as np

class NaiveBayes(object):

    def __init__(self, model=0, laplace=0):
        """
        model=0 represents bernoulli, model=1 represents multinomial.
        """
        self.model = model
        self.laplace = laplace
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.m = 0 # count number of training set
        self.n = None # count number of feature
        self.c = None # Y's type
        self.k = 0 # count number of Y's type
        self.v = None # vocabulary
        self.s = 0 # count number of vocanulary

    def load_training_data(self, filename):
        """
        Y:x1 x2 ... xn
        """
        Y = []
        X = []
        with open(filename, 'r') as fp:
            for line in fp:
                line = line.strip('\n').split(':')
                Y.append(line[0])
                line = line[1].split(' ')
                X.append(line)
                self.m += 1
        if self.model == 0:
            for i in xrange(len(X)):
                X[i] = list(set(X[i]))
        self.c = list(set(Y))
        self.k = len(self.c)
        self.v = list({item for line in X for item in line})
        self.s = len(self.v)
        self.format_data(X, Y)


    def load_test_data(self, filename):
        """
        x1 x2 ... xn
        """
        X = []
        self.test_m = 0
        with open(filename, 'r') as fp:
            for line in fp:
                line = line.strip('\n').split(' ')
                X.append(line)
                self.test_m += 1
        if self.model == 0:
            X = [list(set(item)) for item in X]
            self.test_x = np.mat(np.zeros((self.test_m, self.s)))
            for i in xrange(self.test_m):
                for j in xrange(len(X[i])):
                    for k in xrange(self.s):
                        if X[i][j] == self.v[k]:
                            self.test_x[i, k] = 1
                            break
        else:
            self.test_x = X
            for i in xrange(self.test_m):
                for j in xrange(len(X[i])):
                    for k in xrange(self.s):
                        if X[i][j] == self.v[k]:
                            self.test_x[i][j] = k

    def format_data(self, data_x, data_y):
        if self.model == 0:
            self.train_x = np.mat(np.zeros((self.m, self.s)))
            self.train_y = np.mat(np.zeros((self.m, 1)))
            self.sum_yck = np.mat(np.zeros((self.k, 1)))
            self.sum_xn = np.mat(np.zeros((self.k, self.s)))
            for i in xrange(self.m):

                for j in xrange(self.k):
                    if data_y[i] == self.c[j]:
                        self.train_y[i] = j
                        self.sum_yck[j] += 1
                        break

                for j in xrange(len(data_x[i])):
                    for k in xrange(self.s):
                        if data_x[i][j] == self.v[k]:
                            self.train_x[i, k] = 1
                            self.sum_xn[int(self.train_y[i]), k] += 1
                            break
        else:
            self.train_x = data_x
            self.train_y = data_y
            self.sum_yck = np.mat(np.zeros((self.k, 1)))
            self.sum_xn = np.mat(np.zeros((self.k, self.s)))
            for i in xrange(self.m):

                for j in xrange(self.k):
                    if data_y[i] == self.c[j]:
                        self.train_y[i] = j
                        self.sum_yck[j] += 1
                        break

                for j in xrange(len(data_x[i])):
                    for k in xrange(self.s):
                        if data_x[i][j] == self.v[k]:
                            self.train_x[i][j] = k
                            self.sum_xn[int(self.train_y[i]), k] += 1
                            break

    def bernoulli_learn(self):
        self.prob_t = np.mat(np.zeros((self.k, self.s)))
        self.prob_y = np.mat(np.zeros((self.k, 1)))
        for i in xrange(self.k):
            for j in xrange(self.s):
                self.prob_t[i, j] = (self.sum_xn[i, j] + self.laplace) / (self.sum_yck[i] + self.laplace * 2.0)
            self.laplace = 0.0
            self.prob_y[i] = (self.sum_yck[i] + self.laplace) / (self.m + self.k * self.laplace)
            self.laplace = 1.0

    def bernoulli_predict(self):
        prob = np.mat(np.zeros((self.test_m, self.k)))
        for i in xrange(self.test_m):
            for j in xrange(self.k):
                prob[i, j] = self.prob_y[j]
                for k in xrange(self.s):
                    if self.test_x[i, k] == 1:
                        prob[i, j] *= self.prob_t[j, k]
                    else:
                        prob[i, j] *= (1.0 - self.prob_t[j, k])
        return prob

    def multinomial_learn(self):
        self.prob_t = np.mat(np.zeros((self.k, self.s)))
        self.prob_y = np.mat(np.zeros((self.k, 1)))
        for i in xrange(self.k):
            for j in xrange(self.s):
                self.prob_t[i, j] = (self.sum_xn[i, j] + self.laplace) / (int(self.sum_xn.sum(1)[i]) + self.laplace * self.s)
            self.laplace = 0.0
            self.prob_y[i] = (self.sum_yck[i] + self.laplace) / (self.m + self.k * self.laplace)
            self.laplace = 1.0

    def multinomial_predict(self):
        prob = np.mat(np.zeros((self.test_m, self.k)))
        for i in xrange(self.test_m):
            for j in xrange(self.k):
                prob[i, j] = self.prob_y[j]
                for k in xrange(len(self.test_x[i])):
                    prob[i, j] *= self.prob_t[j, int(self.test_x[i][k])]
        return prob

    def naive_learn(self):
        if self.model == 0:
            self.bernoulli_learn()
        else:
            self.multinomial_learn()

    def naive_predict(self):
        if self.model == 0:
            return self.bernoulli_predict()
        else:
            return self.multinomial_predict()

if __name__ == '__main__':
    nb = NaiveBayes(1, 1.0)
    nb.load_training_data('training.data')
    nb.naive_learn()
    nb.load_test_data('test.data')
    prob = nb.naive_predict()
    print prob
    for ind in prob.argmax(1):
        print nb.c[ind]




