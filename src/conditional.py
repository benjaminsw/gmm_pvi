import jax
import equinox as eqx
import jax.numpy as np
from abc import abstractmethod
from src.nn import XYNet
from jax.scipy.stats import norm, multivariate_normal as mvn
from jax import tree_util as jtu


class Conditional(eqx.Module):
    d_x : int
    d_z : int
    d_y : int

    @abstractmethod
    def log_prob(self, x: jax.Array, z: jax.Array, y: jax.Array):
        raise NotImplementedError()

    @abstractmethod
    def f(self, z: jax.Array, y: jax.Array, eps: jax.Array):
        raise NotImplementedError()
    
    @abstractmethod
    def base_sample(self, key: jax.random.PRNGKey, n_samples: int):
        raise NotImplementedError()

    def get_filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        return filter_spec

    def sample(self, key: jax.random.PRNGKey, n_samples: int, z: jax.Array, y: jax.Array):
        eps = self.base_sample(key, n_samples)
        return self.f(z, y, eps)
    
    def apply_updates(self, update):
        return eqx.apply_updates(self, update)


class FactoredNormCond(Conditional):
    def log_prob(self, x: jax.Array, z: jax.Array, y: jax.Array):
        mu, scale = self.get_mu_scale(z, y)
        return np.sum(norm.logpdf(x, mu, scale), axis=-1)

    def base_sample(self, key: jax.random.PRNGKey, n_samples: int):
        assert type(n_samples) == int
        eps = jax.random.normal(key, (n_samples, self.d_x))
        return eps
    
    def get_mu_scale(self, z: jax.Array, y: jax.Array):
        raise NotImplementedError()

    def f(self, z: jax.Array, y: jax.Array, eps: jax.Array):
        assert z.ndim == 1
        assert y == None or y.ndim == 1
        mu, scale = self.get_mu_scale(z, y)
        out= mu + eps * scale
        return out


class FullNormCond(Conditional):
    def log_prob(self, x: jax.Array, z: jax.Array, y: jax.Array):
        mu, cov = self.get_mu_cov(z, y)
        return mvn.logpdf(x, mu, cov)

    def base_sample(self, key: jax.random.PRNGKey, n_samples: int):
        assert type(n_samples) == int
        eps = jax.random.normal(key, (n_samples, self.d_x))
        return eps
    
    def get_mu_cov(self, z: jax.Array, y: jax.Array):
        raise NotImplementedError()

    def f(self, z: jax.Array, y: jax.Array, eps: jax.Array):
        assert z.ndim == 1
        assert y == None or y.ndim == 1
        mu, cov = self.get_mu_cov(z, y)
        L = np.linalg.cholesky(cov)
        out= mu + eps @ L.T
        return out


class ConstVarNormCond(FactoredNormCond):
    scale : float
    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 scale: float=1.,
                 **kwargs):
        self.scale = np.array(scale)
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y
    
    def get_mu_scale(self, z, y):
        return z, self.scale

    def get_filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.scale,
                                  filter_spec,
                                  True)
        return filter_spec


class DiagNormCond(FactoredNormCond):
    net : eqx.Module

    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        self.net = XYNet(
            key,
            d_x,
            d_z,
            d_y,
            n_hidden,)
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y

    def get_filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.net,
                                  filter_spec,
                                  self.net.get_filter_spec())
        return filter_spec

    def get_mu_scale(self, z, y):
        mu, scale = self.net(z, y)
        return mu, np.maximum(scale, 1e-6)


class FixedDiagNormCondWSkip(FactoredNormCond):
    net : eqx.Module
    sigma : jax.Array

    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        self.net = XYNet(
            key,
            d_x,
            d_z,
            d_y,
            n_hidden)
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y
        #assert self.d_x == d_z
        self.sigma = np.ones(d_x)

    def get_filter_spec(self,):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.net,
                                  filter_spec,
                                  self.net.get_filter_spec())
        filter_spec = eqx.tree_at(lambda tree : tree.sigma,
                                  filter_spec,
                                  jtu.tree_map(eqx.is_array,
                                               self.sigma))
        return filter_spec

    def get_mu_scale(self, z, y):
        mu, _ = self.net(z, y)
        return mu + z, np.maximum(self.sigma,1e-6)


class FixedDiagNormCond(FactoredNormCond):
    net : eqx.Module
    sigma : jax.Array

    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        self.net = XYNet(
            key,
            d_x,
            d_z,
            d_y,
            n_hidden)
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y
        self.sigma = np.ones(d_x)

    def get_filter_spec(self,):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.net,
                                  filter_spec,
                                  self.net.get_filter_spec())
        filter_spec = eqx.tree_at(lambda tree : tree.sigma,
                                  filter_spec,
                                  jtu.tree_map(eqx.is_array,
                                               self.sigma))
        return filter_spec

    def get_mu_scale(self, z, y):
        mu, _ = self.net(z, y)
        return mu, np.maximum(np.exp(self.sigma/2), 1e-6)


class FixedDiagNormCondWLinearSkip(FixedDiagNormCondWSkip):
    linear: eqx.Module
    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        key1, key2 = jax.random.split(key, 2)
        super().__init__(key1,
                         d_x,
                         d_z,
                         d_y,
                         n_hidden)
        self.linear = eqx.nn.Linear(
            d_z,
            d_x,
            key=key2,
            use_bias=False)

    def get_mu_scale(self, z: jax.Array, y: jax.Array):
        mu, _ = self.net(z, y)
        return mu + self.linear(z), np.maximum(self.sigma, 1e-6)


class DiagNormCondWLinearSkip(DiagNormCond):
    linear: eqx.Module
    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        key1, key2 = jax.random.split(key, 2)
        super().__init__(key1,
                         d_x,
                         d_z,
                         d_y,
                         n_hidden)
        self.linear = eqx.nn.Linear(
            d_z,
            d_x,
            key=key2,
            use_bias=False)

    def get_mu_scale(self, z: jax.Array, y: jax.Array):
        mu, scale = super().get_mu_scale(z, y)
        return self.linear(z) + mu, scale


class FixedFullNormCond(FullNormCond):
    net : eqx.Module
    cov : jax.Array
    linear: eqx.Module

    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        self.net = XYNet(
            key,
            d_x,
            d_z,
            d_y,
            n_hidden)
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y
        self.cov = np.eye(d_x)
        self.linear = eqx.nn.Linear(
            d_z,
            d_x,
            key=key,
            use_bias=False)

    def get_filter_spec(self,):
        filter_spec = jtu.tree_map(lambda _: False,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.net,
                                  filter_spec,
                                  self.net.get_filter_spec())
        filter_spec = eqx.tree_at(lambda tree : tree.cov,
                                  filter_spec,
                                  jtu.tree_map(eqx.is_array,
                                               self.cov))
        return filter_spec

    def get_mu_cov(self, z: jax.Array, y: jax.Array):
        mu, _ = self.net(z, y)
        jitter = np.eye(self.d_x) * 1e-8
        cov = (self.cov + self.cov.T) / 2
        return mu + self.linear(z), jax.scipy.linalg.expm(cov) + jitter


KERNELS = {'constant': ConstVarNormCond,
           'norm_var': DiagNormCond,
           'norm_var_lskip': DiagNormCondWLinearSkip,
           'norm_fixed_var_w_lskip': FixedDiagNormCondWLinearSkip,
           'norm_fixed_var': FixedDiagNormCond,
           'norm_fixed_var_w_skip': FixedDiagNormCondWSkip,
           'FixedFullNormCond': FixedFullNormCond}