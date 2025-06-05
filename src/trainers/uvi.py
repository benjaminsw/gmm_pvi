import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from jax.random import split
from src.trainers.util import loss_step
from jaxtyping import PyTree
from src.base import (
    SVICarry,
    UVIParameters,
    Target)


def de_loss(params: PyTree,
            static: PyTree,
            key: jax.random.PRNGKey,
            target: Target,
            y: jax.Array,
            n_samples: int):
    '''
    Density Estimation Loss with path-wise gradients
    '''
    sid = eqx.combine(params, static)
    skey, lqkey = jax.random.split(key, 2)
    _samples, z = sid.sample_joint(skey,
                                   n_samples,
                                   y)
    logq = vmap(eqx.combine(stop_gradient(params), static).elogq,
                (0, 0, 0, None))(split(lqkey, n_samples),
                                 _samples,
                                 z,
                                 y)
    logp = vmap(target.log_prob, (0, None))(_samples, y)
    return np.mean(logq - logp, axis=0)


def de_step(key: jax.random.PRNGKey,
            carry: SVICarry,
            target: Target,
            y: jax.Array,
            optim,
            hyperparams: UVIParameters):
    def loss(key, params, static):
        return de_loss(params,
                       static,
                       key,
                       target,
                       y,
                       n_samples=hyperparams.mc_n_samples)
    lval, id, opt_state = loss_step(
        key,
        loss, 
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state)
    return lval, SVICarry(id, opt_state)