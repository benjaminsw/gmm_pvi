import jax
import jax.numpy as np
from jax import vmap
from jax.random import split
from jax.scipy.special import logsumexp
from src.trainers.util import loss_step
import equinox as eqx
from jaxtyping import PyTree
from src.base import (
    SVICarry,
    SVIOpt,
    SVIParameters,
    Target)


def de_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            n_samples: int,
            K: int):
    sid = eqx.combine(params, static)
    xzkey, zkey = split(key, 2)
    x_sample, z_sample = sid.sample_joint(xzkey,
                                          n_samples,
                                          y)
    zo_samples = vmap(lambda k: sid.sample_base(k, n_samples))(split(zkey, K))
    
    rho = lambda x, z: sid.conditional.log_prob(x,
                                                z,
                                                y)
    vcond = vmap(rho, (0, 0))
    elogqz = np.expand_dims(vcond(x_sample, z_sample), 0)
    vcond = vmap(vcond, (None, 0))
    elogqz = np.append(vcond(x_sample, zo_samples),
                       elogqz,
                       axis=0)
    assert elogqz.shape == (K + 1, n_samples)
    elogqz = logsumexp(elogqz, axis=0) - np.log(K + 1)

    vtarget = vmap(target.log_prob, (0, None))
    elogp = np.mean(vtarget(x_sample, y), axis=0)
    return np.mean(elogqz - elogp, axis=0)


def de_step(key: jax.random.PRNGKey,
            carry: SVICarry,
            target: Target,
            y: jax.Array,
            optim: SVIOpt,
            hyperparams: SVIParameters):
    def loss(key, params, static):
        return de_loss(key,
                       params,
                       static,
                       target,
                       y,
                       n_samples=hyperparams.mc_n_samples,
                       K=hyperparams.K)
    val, model, opt_state = loss_step(key,
                                      loss,
                                      carry.id,
                                      optim.theta_optim,
                                      carry.theta_opt_state)
    return val, SVICarry(id=model,
                         theta_opt_state=opt_state)