"""
Logistic regression for MNIST classification
Niru Maheswaranathan
10-27-2014
"""

import os
import time
import numpy as np

from scipy.io import loadmat
import theano.tensor as T
import theano

class LogisticRegression(object):
    """
    Multi-class logistic regression
    """

    def __init__(self, input, n_in, n_out):
        """
        initializes the logistic regression model
        :param input:
        :param n_in:
        :param n_out:
        :return:
        """

        # initialize the weights W as a zeros matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W', borrow=True)

        # initialize the biases b as a vector of zeros
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership probabilities
        # i.e. p(y | x, W, b) = softmax(Wx + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction from probabilities
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        defines the negative log-likelihood objective
        returns the mean of the neg. LL of the prediction over the data

        :param y:
        :return:
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        returns a float representation of the fraction of errors in the minibatch
        :param y:
        :return:
        """
        return T.mean(T.neg(self.y_pred, y))

def load_data(frac_train, datadir='/data/mldata/', seed=1234):
    """
    loads MNIST data and selects train / test indicse
    :param datadir:
    :param frac_train:
    :return:
    """
    np.random.seed(seed)

    # load data
    mnist = loadmat(os.path.join(datadir, 'mnist-original.mat'))

    # select train / test indices
    num_samples = mnist['label'].size
    num_train = np.round(frac_train * num_samples).astype('int')
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices  = indices[num_train:]

    # return data
    train_data = {'input':   mnist['data'].take(train_indices, axis=1),
                  'labels': mnist['label'].take(train_indices, axis=1)}

    test_data = {'input':   mnist['data'].take(test_indices, axis=1),
                 'labels': mnist['label'].take(test_indices, axis=1)}

    return train_data, test_data

def sgd_mnist(learning_rate=0.1, n_epochs=1000, frac_train=0.8):

    train_data, test_data = load_data(frac_train)

    # x = T.matrix('x')
    # y = T.ivector('y')
    # classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    # cost = classifier.negative_log_likelihood(y)
    #
    # theano.function(
    #     inputs=[index],
    #     outputs=cost,
    #     givens={
    #         x: train_set_x
    #         y: train_set_y
    #     }
    # )
