# Wasserstein Optimizer for R as a Optax.GradientTransformation
import jax
import jax.numpy as np

from optax._src.base import (Schedule,
                             ScalarOrSchedule)
from optax._src.transform import (ScaleByScheduleState)
from optax._src.numerics import safe_int32_increment
from src.base import (GradientTransformation,
                      EmptyState,
                      OptState)
from typing import NamedTuple


class StochasticGradient(NamedTuple):
    drift: jax.numpy.ndarray
    noise: jax.numpy.ndarray


class W2OptState(NamedTuple):
    key: jax.random.PRNGKey


def clip_grad_norm(max_norm: float,
                   per_particle: bool) -> GradientTransformation:
    def init_fn(*args):
        return EmptyState()

    def update_fn(grads: StochasticGradient,
                  state: OptState,
                  params):
        if per_particle:
            norm = np.linalg.norm(grads.drift,
                                  axis=1,
                                  keepdims=True)
        else: # Per dimension
            norm = np.linalg.norm(grads.drift,
                                  axis=0,
                                  keepdims=True)
        ones = np.ones(norm.shape)
        scale = np.where(norm < max_norm,
                         ones,
                         (max_norm / norm))
        return StochasticGradient(drift=scale * grads.drift,
                                  noise=scale ** 0.5 * grads.noise), state
    return GradientTransformation(init_fn, update_fn)


def scale_by_schedule(schedule: Schedule) -> GradientTransformation:
    def init_fn(*args):
        return ScaleByScheduleState(count=np.zeros((), np.int32))

    def update_fn(grads,
                  state: OptState,
                  params,
                  **kwargs):
        lr = schedule(state.count)
        return (StochasticGradient(drift= - lr * grads.drift,
                                   noise=lr ** 0.5 * grads.noise),
                ScaleByScheduleState(count=safe_int32_increment(state.count)))
    return GradientTransformation(init_fn, update_fn)


def regularized_wasserstein_descent(
        key: jax.random.PRNGKey,
        ent_lambda: float) -> GradientTransformation:

    def init_fn(*args):
        return W2OptState(key=key)

    def update_fn(grads,
                  state: W2OptState,
                  params,
                  **kwargs
    ):
        normal_key, key = jax.random.split(state.key, 2)
        eps = jax.random.normal(normal_key, grads.shape)
        drift = grads + ent_lambda * params
        ent_decay = (2 * ent_lambda) ** 0.5 * eps # Entropic Decay
        return  (StochasticGradient(drift=drift,
                                    noise=ent_decay),
                 W2OptState(key))
    return GradientTransformation(init_fn, update_fn)

def stochastic_gradient_to_update() -> GradientTransformation:
    def init_fn(*args):
        return EmptyState()

    def update_fn(grads: StochasticGradient,
                  state: OptState,
                  params,
                  **kwargs):
        return grads.drift + grads.noise, state
    return GradientTransformation(init_fn, update_fn)


def kl_descent(key,) -> GradientTransformation:
    def init_fn(*args):
        return W2OptState(key=key)

    def update_fn(grads,
                  state: OptState,
                  params,
                  **kwargs):
        noise_key, key = jax.random.split(state.key, 2)
        noise = jax.random.normal(noise_key, grads.shape)
        return StochasticGradient(grads,
                                  2 ** 0.5 * noise), W2OptState(key)
    return GradientTransformation(init_fn, update_fn)

def lr_to_schedule(lr: ScalarOrSchedule) -> Schedule:
    if callable(lr):
        return lr
    return lambda _ : lr