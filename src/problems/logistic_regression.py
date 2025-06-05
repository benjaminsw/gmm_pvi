import jax
from scipy.io import loadmat
import jax.numpy as np
import pandas as pd
import jax.debug

from src.model.logistic_regression import LReg


class LogisticRegressionProblem:
    def log_prob(self, x, y=None, train=True):
        assert y is not None
        if train:
            data = self.train
        else:
            data = self.test
        if y is not None:
            subset_data = data[0][y], data[1][y]
        log_likeli_fn = lambda feature, target : self.lr.log_likeli(
            x,
            (target, feature)
        )
        log_likeli =  jax.vmap(log_likeli_fn, (0, 0))(*subset_data) 
        log_prior = self.lr.log_prior(x)
        assert log_prior.shape == log_likeli.shape[1:]
        return log_likeli.mean(0) * data[0].shape[0] + log_prior


class Waveform(LogisticRegressionProblem):
    def __init__(self):
        data = loadmat('./data/waveform/waveform.mat')
        X_train, y_train = data['X_train'], data['y_train'].T
        X_test, y_test = data['X_test'], data['y_test'].T

        '''
        X_cov_train, X_cov_test = standardize_dataset(
            X_train[:, 1:],
            X_test[:, 1:],
        )
        X_train[:, 1:] = X_cov_train
        X_test[:, 1:] = X_cov_test
        print(X_train.mean(0), X_train.std(0))
        '''
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == y_train.shape[0], f"{X_train.shape[0]} != {y_train.shape[0]}"
        assert X_test.shape[0] == y_test.shape[0]

        self.train = (
            np.array(X_train),
            np.array(y_train)
        )
        self.test = (
            np.array(X_test),
            np.array(y_test)
        )
        self.lr = LReg()
        self.train_size = X_train.shape[0]
        self.test_size = X_test.shape[0]
        self.dim = X_train.shape[1]
        self.de = True


def standardize_dataset(X_train,
                        X_test):
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test