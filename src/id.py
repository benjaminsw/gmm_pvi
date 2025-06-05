import jax
from typing import Union
from jax import vmap, grad
import jax.numpy as np
import equinox as eqx
from abc import abstractmethod
import jax.tree_util as jtu
from jax.random import split
from jax.lax import stop_gradient
from src.sampler import hmc

class ID(eqx.Module):
    """
    Implicit Distribution.
    """
    conditional : eqx.Module

    @abstractmethod
    def log_prob(self, x: jax.Array, y: jax.Array):
        raise NotImplementedError()
    
    @abstractmethod
    def sample(self, key: jax.random.PRNGKey, n_samples: int, y: jax.Array):
        raise NotImplementedError()
    
    def get_filter_spec(self,):
        filter_spec = jtu.tree_map(lambda _ : False, self)
        filter_spec = eqx.tree_at(lambda tree : tree.conditional,
                                  filter_spec,
                                  self.conditional.get_filter_spec())
        return filter_spec
    
    def apply_updates(self, updates):
        id, cond = (eqx.apply_updates(self, updates),
                    self.conditional.apply_updates(updates.conditional))
        id = eqx.tree_at(lambda tree : tree.conditional,
                         id,
                         cond)
        return id


class PID(ID):
    """
    Particle Semi-Implicit Distribution.
    """
    particles : jax.numpy.ndarray
    n_particles : int

    def __init__(self,
                 key: jax.random.PRNGKey,
                 conditional: eqx.Module,
                 n_particles: int,
                 init: Union[None, jax.Array] = None):
        self.conditional = conditional
        if init is None:
            init = jax.random.normal(key, (n_particles, conditional.d_z))
        self.particles = init
        assert self.particles.shape == (n_particles, conditional.d_z)
        self.n_particles = n_particles

    def log_prob(self, x: jax.Array, y: jax.Array):
        '''
        Returns the log-probability of q(x|y).
        '''
        fcond = jax.vmap(self.conditional.log_prob,
                         in_axes=(None, 0, None))
        log_prob = fcond(x, self.particles, y)
        assert log_prob.shape == (self.n_particles,), f"Shape of log_prob is {log_prob.shape}"
        return jax.scipy.special.logsumexp(log_prob, axis=0) - np.log(self.n_particles)

    def sample(self,
               key: jax.random.PRNGKey,
               n_samples: int,
               y: jax.Array):
        '''
        Returns samples from the marginal distribution q(x).
        '''
        ckey1, ckey2 = jax.random.split(key)
        sampled_z_ind = jax.random.randint(
            ckey1,
            (n_samples,),
            0,
            self.n_particles)
        sampled_z = self.particles[sampled_z_ind]
        _sample = lambda key, z: self.conditional.sample(key, 1, z, y)
        samples = jax.vmap(_sample, (0, 0))(split(ckey2, n_samples), sampled_z)[:, 0]
        return samples
    

class SID(ID):
    """
    Semi-Implicit Distribution with mixing distribution N(0,I).
    """
    def __init__(self,
                 conditional: eqx.Module):
        self.conditional = conditional

    def log_prob(self,
                 key: jax.random.PRNGKey,
                 x: jax.Array,
                 y: jax.Array,
                 n_samples: int = 100):
        '''
        Returns the log joint probability of q(x, z| y).
        '''
        fcond = jax.vmap(self.conditional.log_prob, in_axes=(None, 0, None))
        z_samples = jax.random.normal(key, (n_samples, self.conditional.d_z))
        log_prob = fcond(x, z_samples, y)
        assert log_prob.shape == (n_samples,), f"Shape of log_prob is {log_prob.shape}"
        return jax.scipy.special.logsumexp(log_prob, axis=0) - np.log(n_samples)
    
    def log_jprob(self, x, z, y):
        '''
        Returns the log joint probability of q(x, z| y).
        '''
        log_likeli = self.conditional.log_prob(x, z, y)
        log_prob = np.sum(jax.scipy.stats.norm.logpdf(z), axis=-1)
        assert log_likeli.shape == log_prob.shape
        return log_likeli + log_prob
    
    def elogq(self,
              key: jax.random.PRNGKey,
              x: jax.Array,
              z: jax.Array,
              y: jax.Array,
              n_samples: int = 10,
              n_steps: int = 5,
              step_size: float = 1e-1):
        '''
        Returns the marginal log-probability of q(x) as an expectation over posterior samples.
        '''
        sampler = lambda key, x: self.psample_hmc(
            key,
            n_samples,
            x,
            z,
            y,
            n_steps,
            step_size)
        psamples = sampler(key, x)
        vcond = vmap(self.conditional.log_prob, (None, 0, None))
        return np.mean(vcond(x, stop_gradient(psamples), y), axis=0)
    
    def psample_hmc(self, key: jax.random.PRNGKey,
                    n_samples: int,
                    x: jax.Array,
                    z: jax.Array,
                    y: jax.Array,
                    L: int,
                    step_size: float,
                    burn_in: int = 50):
        log_prob = lambda z : self.log_jprob(x, z, y)
        sampler_f = lambda key, z: hmc(key,
                                       z,
                                       log_prob,
                                       n_samples,
                                       step_size,
                                       L,
                                       burn_in)
        return sampler_f(key, z)
        
    def sample(self, key: jax.random.PRNGKey, n_samples: int, y: jax.Array):
        return self.sample_joint(key, n_samples, y)[0]

    def sample_base(self, key: jax.random.PRNGKey, n_samples: int):
        sampled_z = jax.random.normal(key, (n_samples, self.conditional.d_z))
        return sampled_z

    def sample_joint(self, key: jax.random.PRNGKey, n_samples: int, y: jax.Array):
        ckey1, ckey2 = jax.random.split(key)
        sampled_z = self.sample_base(ckey1, n_samples)
        _sample = lambda key, z: self.conditional.sample(key, 1, z, y)
        samples = jax.vmap(_sample, (0, 0))(split(ckey2, n_samples), sampled_z)[:, 0]
        return samples, sampled_z


if __name__ == "__main__":
    from src.conditional import NormConditional
    key = jax.random.PRNGKey(0)
    cond = NormConditional(key, 2, 2, 0, 2)
    pid = PID(key, 2, cond, 100)
    pid.sample(key, 100, None)