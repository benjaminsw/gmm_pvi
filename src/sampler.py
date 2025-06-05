import jax
from jax.random import split
import jax.numpy as np
from typing import Callable


def leapfrog(x: jax.Array,
             m: jax.Array,
             log_prob_fn: Callable,
             step_size: float,
             n_steps: int):
    """
    Approximates Hamiltonian dynamics using the leapfrog algorithm.
    Adapted from: https://github.com/martin-marek/mini-hmc-jax/blob/main/hmc.py
    """
    # define a single step
    def step(xm, i):
        x, m = xm
        
        # update momentum
        grad = jax.grad(log_prob_fn)(x)
        m += 0.5 * step_size * grad

        # update params
        x += m * step_size

        # update momentum
        grad = jax.grad(log_prob_fn)(x)
        m += 0.5 * step_size * grad
        return (x, m), None

    # do 'n_steps'
    xm0 = (x, m)
    new_xm, _ = jax.lax.scan(step, xm0, np.arange(n_steps))
    return new_xm

def hmc(key: jax.random.PRNGKey,
        x0: jax.Array,
        log_prob_fn: Callable,
        n_samples: int,
        step_size: float,
        n_leapfrog_steps: int,
        burn_in: int):
    """
    HMC Sampler.
    Adapted from: https://github.com/martin-marek/mini-hmc-jax/blob/main/hmc.py
    """
    n_steps = burn_in + n_samples
    def step(x_step, key):
        x, step_size = x_step
        mkey, akey = jax.random.split(key, 2)
        m = jax.random.normal(mkey, x.shape) # Generate Momentum
        new_x, new_m = leapfrog(x, m, log_prob_fn, step_size, n_leapfrog_steps) # Leapfrog 

        # Metropolis-Hastings Correction
        logp_diff = log_prob_fn(new_x) - log_prob_fn(x)
        potential = 0.5 * (m ** 2 - new_m ** 2).sum(-1)
        assert potential.shape == (), "Potential shape is {}".format(potential.shape)
        assert logp_diff.shape == potential.shape
        accept_prob = np.minimum(1, np.exp(logp_diff + potential))
        accept = jax.random.uniform(akey) < accept_prob
        x = np.where(accept, new_x, x)
        # Adapt step size according to 
        # https://github.com/franrruiz/uivi/blob/master/src/mcmc/hmc.m
        new_step_size = step_size + 0.01 * (accept - 0.9) / 0.9 * step_size
        return (x, new_step_size), x
    _, path = jax.lax.scan(step,
                           (x0, step_size),
                           split(key, n_steps))
    return path[burn_in:]

