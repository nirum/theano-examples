"""
Logistic regression for MNIST classification
Niru Maheswaranathan
10-27-2014
"""

import os
from time import time
import numpy as np
from scipy.io import loadmat

import theano.tensor as T
import theano

from sfo.sfo import SFO

class LogisticRegression(object):

    def __init__(self, train_data, test_data):

        # data is a list of minibatches, each element of which is a dictionary with x and y keys
        # x: num_features, num_samples
        # y: num_samples,
        self.train = train_data
        self.test  = test_data

        # initialize parameters
        self.num_classes = np.unique(np.hstack([t['y'] for t in self.train])).size
        self.num_features = train_data[0]['x'].shape[1]
        self.minibatch_size = train_data[0]['y'].size
        self.theta_init = 1e-4*np.random.randn(self.num_features, self.num_classes).ravel()

        # optimize using SFO
        self.optimizer = SFO(self.f_df_wrapper, self.theta_init, self.train, display=2)

        # theano variables
        tx = T.matrix('x')
        tw = T.matrix('theta')
        ty = T.ivector('y')

        # negative log-likelihood objective
        self.p_y_given_x = T.nnet.softmax(tx.dot(tw))
        self.y_pred = theano.function([tx,tw], T.argmax(self.p_y_given_x, axis=1))
        self.loss = -T.mean(T.log(self.p_y_given_x)[np.arange(self.minibatch_size), ty])
        self.f_df = theano.function([tx, ty, tw], [self.loss, T.grad(self.loss, tw)])

    def f_df_wrapper(self, theta, data):
        """
        wrap the symbolic f_df function, passing in data from a single minibatch
        """
        theta_matrix = theta.reshape(self.num_features, self.num_classes)
        f, df = self.f_df(data['x'], data['y'], theta_matrix)
        return f, df.ravel()

    def fit(self, num_passes=5):
        """
        Optimize the logistic regression model using SFO
        """

        # fit
        theta_vector = self.optimizer.optimize(num_passes=num_passes)
        self.theta = theta_vector.reshape(self.num_features, self.num_classes)

        # test
        frac_correct = self.classify(self.theta)
        print('---------------------------------------------')
        print('--- Fraction correct on test set: %5.4f ---' % frac_correct)
        print('---------------------------------------------')

        return frac_correct

    def classify(self, theta):
        ypred = np.hstack([self.y_pred(t['x'], theta) for t in self.test])
        ytrue = np.hstack([t['y'] for t in self.test])
        frac_correct = np.mean(ypred-ytrue == 0)
        return frac_correct

def load_data(minibatch_size=100, frac_train=0.8, datadir='~/data/mldata/'):
    """
    loads MNIST data and selects train / test indices
    :param datadir:
    :param frac_train:
    :return:
    """

    # load data
    mnist = loadmat(os.path.join(os.path.expanduser(datadir), 'mnist-original.mat'))

    # select random train / test indices
    num_samples = mnist['label'].size
    num_train = np.round(frac_train * num_samples).astype('int')
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # comprehension to generate train and test data set
    train_data = [{'x': np.vstack((mnist['data'].take(idx,axis=1), np.ones((1,minibatch_size)))).T,
                   'y': mnist['label'].take(idx,axis=1).ravel().astype('int32')}
                  for idx in indices[:num_train].reshape(-1, minibatch_size)]

    test_data  = [{'x': np.vstack((mnist['data'].take(idx,axis=1), np.ones((1,minibatch_size)))).T,
                   'y': mnist['label'].take(idx,axis=1).ravel().astype('int32')}
                  for idx in indices[num_train:].reshape(-1, minibatch_size)]

    return train_data, test_data

if __name__ == '__main__':

    # set random seed
    np.random.seed(1234)

    # load train / test data, split into minibatches
    train, test = load_data()

    #