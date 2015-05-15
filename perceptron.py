#!/usr/bin/env python
# encoding: utf-8


class Perceptron(object):

    def __init__(self, rate, w0, b0, data):
        """
        rate is learing rate; w0 is the initial value of weight;
        b0 is the initial value of bias; data is training data set.
        """
        self.rate = rate
        self.w = w0
        self.b = b0
        self.data = data

    def is_error_pot(self, x):
        ans = (self.mul_vecter(self.w, x[:-1]) + self.b) * x[-1]
        if ans <= 0:
            return True
        else:
            return False

    def adjust(self, x):
        """
        update weight vecter w and bias b
        """
        self.w = self.add_vecter(self.w, [self.rate * x[-1] * x[i] for i in range(len(x) - 1)] )
        self.b = self.b + self.rate * x[-1]

    def train(self):
        flag = True
        cnt = 0
        print '-' * 30
        while flag:
            for i in range(len(self.data)):
                if self.is_error_pot(self.data[i]):
                    cnt += 1
                    print 'No.{0} adjustment.'.format(cnt)
                    print 'data:', self.data[i]
                    print 'w = ', self.w, ' b = ', self.b
                    self.adjust(self.data[i])
                    flag = True
                    break
                else:
                    flag = False
        return self.w, self.b, cnt

    @staticmethod
    def add_vecter(x, y):
        """
        calculate the sum of two vecter, return a new vecter
        """
        if len(x) != len(y):
            raise Exception
        else:
            return [x[i] + y[i] for i in range(len(x))]

    @staticmethod
    def mul_vecter(x, y):
        """
        calculate the product of two vecter, return a constant
        """
        if len(x) != len(y):
            raise Exception
        else:
            return sum([x[i] * y[i] for i in range(len(x))])


if __name__ == '__main__':
    data = [[3, 3, 1], [4, 3, 1], [1, 1, -1]]
    p = Perceptron(1, [0, 0], 0, data)
    ans = p.train()
    print '-' * 30
    print ans


