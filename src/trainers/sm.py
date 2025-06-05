import jax
from jax import grad, vmap
import jax.numpy as np
import equinox as eqx
from src.trainers.util import loss_step
from src.base import (
    Target,
    SMCarry,
    SMParameters,
    SMOpt,
)
import jax.debug as jdb


def _loss(
        id: eqx.Module,
        dual: eqx.Module,
        target: Target,
        eps: jax.Array,
        z: jax.Array,
        y: jax.Array,
):
    x = id.conditional.f(z, y, eps)
    f = dual(x)
    t_score = lambda x : grad(target.log_prob)(x, y)
    cm_score = lambda x : grad(id.conditional.log_prob)(x, z, y)
    # see https://github.com/longinYu/SIVISM/blob/main/sivism_bnn.py 
    # who utilized these tricks.
    g = t_score(x) + f
    dscore = 2. * t_score(x) - 2. * cm_score(x) - g
    assert f.shape == dscore.shape
    return np.sum(g * dscore, axis=-1)


def de_loss(key: jax.random.PRNGKey,
            sid: eqx.Module,
            dual: eqx.Module,
            target: Target,
            y: jax.Array,
            n_samples: int):
    eps_key, z_key = jax.random.split(key, 2)
    eps = sid.conditional.base_sample(eps_key, n_samples)
    z = sid.sample_base(z_key, n_samples) 
    loss = lambda eps, z: _loss(sid,
                                dual,
                                target,
                                eps,
                                z,
                                y)
    lval = np.mean(vmap(loss, (0, 0))(eps, z), axis=0)
    return lval


def de_theta_step(key: jax.random.PRNGKey,
                  sid: eqx.Module,
                  dual: eqx.Module,
                  target: Target,
                  y: jax.Array,
                  optim,
                  opt_state):
    def theta_loss(key, param, static):
        id = eqx.combine(param, static)
        return de_loss(key,
                       id,
                       dual,
                       target,
                       y,
                       n_samples=250)
    return  loss_step(key,
                      theta_loss,
                      sid,
                      optim,
                      opt_state)


def de_dual_step(key: jax.random.PRNGKey,
                 sid: eqx.Module,
                 dual: eqx.Module,
                 target: Target,
                 y: jax.Array,
                 optim,
                 dual_opt_state):
    def dual_loss(key, param, static):
        dual = eqx.combine(param, static)
        return - de_loss(key,
                         sid,
                         dual,
                         target,
                         y,
                         n_samples=250)
    return loss_step(key,
                     dual_loss,
                     dual,
                     optim,
                     dual_opt_state)


def de_step(key: jax.random.PRNGKey,
            carry: SMCarry,
            target: Target,
            y: jax.Array,
            optim: SMOpt,
            hyperparams: SMParameters,
):
    dual, dual_opt_state = carry.dual, carry.dual_opt_state
    id, theta_opt_state = carry.id, carry.theta_opt_state
    id_key, dual_key = jax.random.split(key, 2)
    iter_keys = jax.random.split(id_key, hyperparams.train_steps)
    for i in range(hyperparams.train_steps):
        lval, id, theta_opt_state = de_theta_step(
            iter_keys[i],
            id,
            dual,
            target,
            y,
            optim.theta_optim,
            theta_opt_state
        )
    iter_keys = jax.random.split(dual_key, hyperparams.dual_steps)
    for i in range(hyperparams.dual_steps):
        _, dual, dual_opt_state = de_dual_step(
            iter_keys[i],
            id,
            dual,
            target,
            y,
            optim.dual_optim,
            dual_opt_state,
    )
    return lval, SMCarry(id=id,
                         theta_opt_state=theta_opt_state,
                         dual=dual,
                         dual_opt_state=dual_opt_state)