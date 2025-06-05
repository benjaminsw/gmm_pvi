import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map


def de_particle_grad(key: jax.random.PRNGKey,
                     pid : PID,
                     target: Target,
                     particles: jax.Array,
                     y: jax.Array,
                     mc_n_samples: int):
    '''
    Compute the gradient of the first variation
    using pathwise monte carlo gradient estimation
    with number of samples set to `mc_n_samples`
    '''
    def ediff_score(particle, eps):
        '''
        Compute the expectation of the difference
        of scores using the reparameterization trick.
        '''
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        assert samples.shape == (mc_n_samples, target.dim)
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        assert logp.shape == (mc_n_samples,)
        assert logq.shape == (mc_n_samples,)
        logp = np.mean(logp, 0)
        logq = np.mean(logq, 0)
        return logq - logp
    eps = pid.conditional.base_sample(key, mc_n_samples)
    # Can replace with map to reduce space requirements
    # grad = map(jax.grad(lambda p: ediff_score(p, eps)), particles)
    grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
    return grad


def de_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            hyperparams: PIDParameters):
    '''
    Density Estimation Loss with path-wise gradients
    '''
    pid = eqx.combine(params, static)
    _samples = pid.sample(key, hyperparams.mc_n_samples, None)
    logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
    logp = vmap(target.log_prob, (0, None))(_samples, y)
    return np.mean(logq - logp, axis=0)


def de_particle_step(key: jax.random.PRNGKey,
                     pid: PID,
                     target: Target,
                     y: jax.Array,
                     optim: PIDOpt,
                     carry: PIDCarry,
                     hyperparams: PIDParameters):
    '''
    Particle Step for Density Estimation.
    '''
    grad_fn = lambda particles: de_particle_grad(
        key,
        pid,
        target,
        particles,
        y,
        hyperparams.mc_n_samples)
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        grad_fn,
        carry.r_precon_state,)
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y)
    pid = eqx.tree_at(lambda tree : tree.particles,
                      pid,
                      pid.particles + update)
    carry = PIDCarry(
        id=pid,
        theta_opt_state=carry.theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state)
    return pid, carry


def de_step(key: jax.random.PRNGKey,
            carry: PIDCarry,
            target: Target,
            y: jax.Array,
            optim: PIDOpt,
            hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    '''
    Density Estimation Step.
    '''
    theta_key, r_key = jax.random.split(key, 2)
    def loss(key, params, static):
        return de_loss(key,
                       params,
                       static,
                       target,
                       y,
                       hyperparams)
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )

    pid, carry = de_particle_step(
        r_key,
        pid,
        target,
        y,
        optim,
        carry,
        hyperparams)

    carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state)
    return lval, carry