import jax
import equinox as eqx
import src.id as ID
from typing import Callable
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray as PRNGKey, PyTree
from operator import add
from jax import numpy as np
from src.base import GradientTransformation, OptState

def loss_step(key: PRNGKey,
              loss: Callable[[PRNGKey, PyTree, PyTree],
                              float],
              model: eqx.Module,
              optim: GradientTransformation,
              opt_state: OptState):
    """
    Loss Step.
    loss : Callable[[PRNGKey, Module], float]
    """
    params, static = eqx.partition(model,
                                   model.get_filter_spec())
    val, grad = jax.value_and_grad(loss, argnums=1)(
        key,
        params,
        static
    )
    updates, opt_state = optim.update(grad,
                                      opt_state,
                                      model)
    # model = model.apply_updates(updates)
    model = eqx.apply_updates(model, updates)
    return val, model, opt_state

def clip_norm(x, max_norm, eps=1e-6, return_norm=False):
    norm = np.linalg.norm
    clip = lambda x: np.where(norm(x) < max_norm, x, x * max_norm / np.maximum(norm(x), eps))
    x_norm = jax.vmap(norm, in_axes=0)(x)
    normed_x = jax.vmap(clip, in_axes=0)(x)
    if return_norm:
        return normed_x, x_norm
    return normed_x
