from sklearn.datasets import fetch_openml
from src.model.bnn import BNN
import jax.numpy as np
import jax
from sklearn.model_selection import train_test_split
import pandas as pd
  

class BNN_Regression_Problem:
    def log_prob(self, x, y=None, train=True):
        assert y is not None
        if train:
            data = self.train
        else:
            data = self.test
        if y is not None:
            subset_data = data[0][y], data[1][y]
        log_likeli_fn = lambda feature, target : self.bnn.log_likeli(x,
                                                              (target, feature))
        log_likeli =  jax.vmap(log_likeli_fn, (0, 0))(*subset_data).mean(0) * data[0].shape[0]
        log_prior = self.bnn.log_prior(x)
        assert log_prior.shape == log_likeli.shape[1:]
        return log_likeli + log_prior
    
    def mse(self, x, train=False):
        if train:
            data = self.train
        else:
            data = self.test
        loc, y = data
        y_model = self.bnn.eval_f(x, loc)
        assert y.ndim == 2
        assert y.shape == y_model.shape
        return np.mean(np.sum((y - y_model) ** 2, axis=-1))


def standardize_dataset(X_train,
                        X_test,
                        Y_train,
                        Y_test):
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    Y_mean, Y_std = np.mean(Y_train, axis=0), np.std(Y_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    Y_train = (Y_train - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std
    return X_train, X_test, Y_train, Y_test


class Concrete(BNN_Regression_Problem):
    """
    https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    """
    def __init__(self,
                 seed=42,
                 test_size=0.2,
                 standardize=True,
                 n_hidden=10):
        data = pd.read_excel("./data/concrete/Concrete_Data.xls")
        target_column = 'Concrete compressive strength(MPa, megapascals) '
        
        X = data.loc[:, data.columns != target_column]
        y = data.loc[:, target_column]

        X_train, X_test, Y_train, Y_test  = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        to_array = lambda x : np.asarray(x.to_numpy())
        X_train = to_array(X_train)
        X_test = to_array(X_test)
        Y_train = np.expand_dims(to_array(Y_train), axis=-1)
        Y_test = np.expand_dims(to_array(Y_test), axis=-1)

        if standardize:
            X_train, X_test, Y_train, Y_test = standardize_dataset(
                X_train,
                X_test,
                Y_train,
                Y_test
            )

        self.train = (
            X_train,
            Y_train,
        )
        self.test = (
            X_test,
            Y_test,
        )
        self.train_size = self.train[0].shape[0]
        self.test_size = self.test[0].shape[0]

        self.de = True

        scale = 0.01 if standardize else 1.
        self.bnn = BNN(
            self.train[0].shape[1],
            self.train[1].shape[1],
            n_hidden,
            scale = scale,
        )
        self.dim = self.bnn.dim
        print("Concrete dataset loaded.")
        print("Train size: ", self.train_size)
        print("Test size: ", self.test_size)
        print("Dimension: ", self.dim)
        print("Number of train features", self.train[0].shape[1])
        print("Number of out", self.train[1].shape[1])


class Yacht(BNN_Regression_Problem):
    def __init__(self,
                 seed=42,
                 test_size=0.2,
                 standardize=True,
                 n_hidden=10):
        data = pd.read_fwf("./data/yacht/yacht_hydrodynamics.data",
                           header=None)
        target_column = 6

        X = data.loc[:, data.columns != target_column]
        y = data.loc[:, target_column]

        X_train, X_test, Y_train, Y_test  = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        to_array = lambda x : np.asarray(x.to_numpy())
        X_train = to_array(X_train)
        X_test = to_array(X_test)
        Y_train = np.expand_dims(to_array(Y_train), axis=-1)
        Y_test = np.expand_dims(to_array(Y_test), axis=-1)

        if standardize:
            X_train, X_test, Y_train, Y_test = standardize_dataset(
                X_train,
                X_test,
                Y_train,
                Y_test
            )

        self.train = (
            X_train,
            Y_train,
        )
        self.test = (
            X_test,
            Y_test,
        )
        self.train_size = self.train[0].shape[0]
        self.test_size = self.test[0].shape[0]
        self.de = True

        scale = 0.01 if standardize else 1.
        self.bnn = BNN(
            self.train[0].shape[1],
            self.train[1].shape[1],
            n_hidden,
            scale = scale,
        )
        self.dim = self.bnn.dim
        print("Yacht dataset loaded.")
        print("Train size: ", self.train_size)
        print("Test size: ", self.test_size)
        print("Dimension: ", self.dim)
        print("Number of train features", self.train[0].shape[1])
        print("Number of out", self.train[1].shape[1])


class Protein(BNN_Regression_Problem):
    """
    https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    """
    def __init__(self,
                 standardize=True,
                 seed=42,
                 n_hidden=30,
                 n_samples=2000,
                 test_size=0.2):
        data = pd.read_csv("./data/protein/CASP.csv")
        target_column = 'RMSD'

        X = data.loc[:n_samples, data.columns != target_column]
        y = data.loc[:n_samples, target_column]

        for column in X.columns:
            X.loc[:, column] = X.loc[:, column].astype(float)

        to_array = lambda x : np.asarray(x.to_numpy())
        X, y = to_array(X), np.expand_dims(to_array(y), axis=-1)

        X_train, X_test, Y_train, Y_test  = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        if standardize:
            X_train, X_test, Y_train, Y_test = standardize_dataset(
                X_train,
                X_test,
                Y_train,
                Y_test
            )

        self.train = (
            X_train,
            Y_train,
        )
        self.test = (
            X_test,
            Y_test,
        )
        self.train_size = self.train[0].shape[0]
        self.test_size = self.test[0].shape[0]

        self.de = True

        scale = 0.01 if standardize else 1.
        self.bnn = BNN(
            self.train[0].shape[1],
            self.train[1].shape[1],
            n_hidden,
            scale = scale,
        )
        self.dim = self.bnn.dim
        print("Protein dataset loaded.")
        print("Train size: ", self.train_size)
        print("Test size: ", self.test_size)
        print("Dimension: ", self.dim)
        print("Number of train features", self.train[0].shape[1])
        print("Number of out", self.train[1].shape[1])