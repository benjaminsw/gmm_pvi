import jax.numpy as np
from jax.scipy.stats import norm
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp
from jaxtyping import Float, Array
from math import pi
import jax
from src.base import Target


class Banana(Target):
    def __init__(self):
        self.dim = 2
        self.de = True
    
    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        z1, z2 = x[..., 0], x[..., 1]
        logpz1 = norm.logpdf(z1,
                            loc=0,
                            scale=2.)
        logpz2 = norm.logpdf(z2,
                            loc=z1 ** 2 / 4,
                            scale=1)
        return logpz1 + logpz2
    
    def sample(self,
               key,
               n_samples,
               y):
        key1, key2 = jax.random.split(key, 2)
        z1 = jax.random.normal(key1, (n_samples, 1)) * 2
        z2 = jax.random.normal(key2, (n_samples, 1))
        x = np.concatenate([z1, z1 ** 2 / 4 + z2], axis=-1)
        return x


class Multimodal(Target):
    def __init__(self):
        self.dim = 2
        self.de = True
        mu1 = np.zeros(2) + 2
        mu2 = np.zeros(2) - 2
        mu3 = np.zeros(2) 
        mu3 = mu3.at[0].set(2)
        mu3 = mu3.at[1].set(-2)
        mu4 = np.zeros(2) 
        mu4 = mu4.at[0].set(-2)
        mu4 = mu4.at[1].set(2)
        self.mu = np.stack([mu1, mu2, mu3, mu4], axis=0)
        self.cov = np.eye(2)
        self.p = np.array([1/8, 1/8, 1/2, 1/4])
    
    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        logp1 = mvn.logpdf(x, self.mu[0], self.cov) + np.log(self.p[0])
        logp2 = mvn.logpdf(x, self.mu[1], self.cov) + np.log(self.p[1])
        logp3 = mvn.logpdf(x, self.mu[2], self.cov) + np.log(self.p[2])
        logp4 = mvn.logpdf(x, self.mu[3], self.cov) + np.log(self.p[3])
        logps = np.stack([logp1, logp2, logp3, logp4], axis=0)
        return logsumexp(logps, axis=0)
    
    def sample(self,
               key,
               n_samples,
               y):
        eps = jax.random.normal(key, (n_samples, 2))
        ind = jax.random.choice(key, 4, shape=(n_samples,), p=self.p)
        eps = eps + self.mu[ind]
        return eps


class Quadmodal(Target):
    def __init__(self, mu):
        self.dim = 2
        self.de = True
        mu1 = np.zeros(2) + mu
        mu2 = np.zeros(2) - mu
        mu3 = np.zeros(2) 
        mu3 = mu3.at[0].set(mu)
        mu3 = mu3.at[1].set(-mu)
        mu4 = np.zeros(2) 
        mu4 = mu4.at[0].set(-mu)
        mu4 = mu4.at[1].set(mu)
        self.mu = np.stack([mu1, mu2, mu3, mu4], axis=0)
        self.cov = np.eye(2)
    
    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        logp1 = mvn.logpdf(x, self.mu[0], self.cov) 
        logp2 = mvn.logpdf(x, self.mu[1], self.cov) 
        logp3 = mvn.logpdf(x, self.mu[2], self.cov) 
        logp4 = mvn.logpdf(x, self.mu[3], self.cov) 
        logps = np.stack([logp1, logp2, logp3, logp4], axis=0)
        return logsumexp(logps, axis=0) - np.log(4)
    
    def sample(self,
               key,
               n_samples,
               y):
        eps = jax.random.normal(key, (n_samples, 2))
        ind = jax.random.choice(key, 4, shape=(n_samples,), p=self.p)
        eps = eps + self.mu[ind]
        return eps
class XShape(Target):
    def __init__(self):
        self.dim = 2
        self.de = True
        self.cov1 = np.array([[2, 1.8],
                        [1.8, 2]])
        self.cov2 = np.array([[2, -1.8],
                        [-1.8, 2]])
        self.cov1_L = np.linalg.cholesky(self.cov1)
        self.cov2_L = np.linalg.cholesky(self.cov2)

    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        logp1 = mvn.logpdf(x, np.zeros(2), self.cov1)
        logp2 = mvn.logpdf(x, np.zeros(2), self.cov2)
        return logsumexp(np.stack([logp1, logp2], axis=0), axis=0) - np.log(2)
    
    def sample(self,
               key,
               n_samples,
               y):
        key1, key2 = jax.random.split(key, 2)
        x1 = np.einsum("ij, ...j-> ...i",
                       self.cov1_L,
                       jax.random.normal(key1, (n_samples, 2)))
        x2 = np.einsum("ij, ...j-> ...i",
                       self.cov2_L,
                       jax.random.normal(key2, (n_samples, 2)))
        x = np.zeros((n_samples, 2))
        ind = jax.random.randint(key, (n_samples,), 0, 2)
        x = x.at[ind == 0, :].set(x1[ind == 0, :])
        x = x.at[ind == 1, :].set(x2[ind == 1, :])
        return x


class Bimodal(Target):
    def __init__(self, mu):
        self.dim = 2
        self.de = True
        self.cov = np.eye(2)
        self.mu1 = np.zeros(2) + mu
        self.mu2 = np.zeros(2) - mu
        self.mu = np.stack([self.mu1, self.mu2], axis=0)
    
    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        logp1 = mvn.logpdf(x, self.mu1, self.cov) 
        logp2 = mvn.logpdf(x, self.mu2, self.cov) 
        logps = np.stack([logp1, logp2], axis=0)
        return logsumexp(logps, axis=0) - np.log(2)
    
    def sample(self,
               key,
               n_samples,
               y):
        eps = jax.random.normal(key, (n_samples, 2))
        ind = jax.random.randint(key, (n_samples,), 0, 2)
        eps = eps + self.mu[ind]
        return eps


class Simple(Target):
    def __init__(self):
        self.dim = 2
        self.de = True

    def log_prob(self,
                 x: Float[Array, "dim"],
                 y):
        cov1 = np.array([[2, 1.8],
                        [1.8, 2]])
        logp1 = mvn.logpdf(x, np.zeros(2) + 2, cov1)
        return logp1


class MultimodalPosterior(Target):
    def __init__(self, sigma: float = 1.):
        self.d_y = 2
        self.dim = 1
        self.de = False
        self.fa = True
        self.sigma = sigma

    def log_prob(self,
                 x,
                 y,):
        assert x.ndim == 1
        assert y.ndim == 1
        log_posterior = self.posterior(x, y)
        logpy = norm.logpdf(y, loc=0, scale=1).sum(-1)
        return log_posterior + logpy
    
    def posterior(self,
                  x,
                  y):
        assert x.ndim == 1
        assert y.ndim == 1
        mu = np.clip(y, -5, 5)
        logpxIy1 = norm.logpdf(x,
                               loc=mu[0],
                               scale=self.sigma).sum(-1)
        logpxIy2 = norm.logpdf(x,
                               loc=mu[1],
                               scale=self.sigma).sum(-1)
        logp = np.stack([logpxIy1, logpxIy2], axis=-1)
        return logsumexp(logp, axis=-1) - np.log(2)
    
    def post_cdf(self,
                 x,
                 y):
        assert x.ndim == 1
        assert y.ndim == 1
        mu = np.clip(y, -5, 5)
        cdf1 = norm.cdf(x,
                        loc=mu[0],
                        scale=self.sigma)
        cdf2 = norm.cdf(x,
                        loc=mu[1],
                        scale=self.sigma)
        return np.squeeze((cdf1 + cdf2)/2, axis=-1)


class CircleInference(Target):
    def __init__(self,
                 eta: float = 0.75,
                 sigma: float = 1.):
        self.de = False
        self.fa = True
        self.dim = 1
        self.d_y = 2
        self.eta = eta
        self.sigma = sigma
    
    def log_prob(self,
                 x: Float[Array, "dim"],
                 y: Float[Array, "d_y"]):
        assert x.ndim == 1
        assert y.ndim == 1
        mean = np.stack([np.sin(pi * np.tanh(self.eta * x)),
                         np.cos(pi * np.tanh(self.eta * x))], axis=-1)[0]
        logpy1Ix = norm.logpdf(y[0], loc=mean[0], scale=self.sigma)
        logpy2Ix = norm.logpdf(y[1], loc=mean[1], scale=self.sigma)
        logpx = norm.logpdf(x, loc=0, scale=1)
        logp = (logpy1Ix + logpy2Ix + logpx)[0]
        assert logp.shape == ()
        return logp
