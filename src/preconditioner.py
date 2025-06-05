import jax
import jax.numpy as np
from typing import Callable, NamedTuple
from src.base import EmptyState


class Preconditioner(NamedTuple):
    init: Callable
    update: Callable


def identity() -> Preconditioner:
    def init_fn(*args):
        return EmptyState()
    
    def update_fn(particles: jax.Array, # Array n_particles x d_z
                  grad_fn: Callable, # Signature Array n_particles x d_z -> Array n_particles x d_z
                  state: EmptyState): # EmptyState
        return grad_fn(particles), EmptyState()
    return Preconditioner(init_fn, update_fn)


def clip_grad_norm(max_norm: int,
              agg_mode: str) -> Preconditioner:
    agg = np.max if agg_mode == 'max' else np.mean

    def init_fn(*args):
        return EmptyState()
    
    def clipping(grad_fn: Callable, # Signature Array n_particles x d_z -> Array n_particles x d_z
                 particles: jax.Array): # Array n_particles x d_z
        norm = np.linalg.norm(grad_fn(particles),
                              axis=1) # Array n_particles
        max = agg(norm)
        clip = np.where(max < max_norm,
                        1,
                        max_norm / max)
        return clip * np.ones(particles.shape[-1])

    def update_fn(particles: jax.Array,
                  grad_fn: Callable,
                  state: EmptyState):
        grad = grad_fn(particles)
        g = clipping(grad_fn,
                     particles)
        if False: 
            ## This computes the additional correction term in the preconditioner update.
            m = jax.jacfwd(lambda x: clipping(grad_fn, x))(particles)
            correction = np.sum(m, axis=-1).T
        else:
            correction = np.zeros(particles.shape)
        assert g.shape == (particles.shape[-1],), f"Shape of g is {g.shape} should be {particles.shape[0]}"
        assert correction.shape == particles.shape, f"Shape of correction is {correction.shape} should be {particles.shape}"
        return np.einsum("j,ij->ij", g, grad) - correction, EmptyState()
    return Preconditioner(init_fn, update_fn)


class RMSState(NamedTuple):
    mean_sq: jax.numpy.ndarray

def rms(agg_mode: str,
        alpha: float=0.9) -> Preconditioner:
    agg = np.max if agg_mode == 'max' else np.mean

    def init_fn(id):
        return RMSState(np.zeros(id.particles.shape[-1]))
    
    def clipping(grad_fn: Callable, # Signature Array n_particles x d_z -> Array n_particles x d_z
                 particles: jax.Array,
                 state: RMSState): # Array n_particles x d_z
        grad = grad_fn(particles)
        grad2 = agg(grad ** 2, axis=0)
        moving_avg = state.mean_sq * alpha + (1 - alpha) * grad2
        g = 1 / np.maximum(moving_avg ** 0.5, 1e-8)
        return g, RMSState(moving_avg)

    def update_fn(particles: jax.Array,
                  grad_fn: Callable,
                  state: RMSState):
        grad = grad_fn(particles)
        g, new_state = clipping(
            grad_fn,
            particles,
            state)
        if False: 
            ## This computes the additional correction term in the preconditioner update.
            _, jvp_fn = jax.linearize(lambda x: clipping(grad_fn, x, state)[0],
                                      particles,
                                      has_aux=False)
            ones = np.zeros((particles.shape[0], *particles.shape))
            for i in range(particles.shape[0]):
                ones = ones.at[i, i,:].set(1.)
            correction = jax.vmap(jvp_fn)(ones)
        else:
            correction = np.zeros(particles.shape)
        assert g.shape == (particles.shape[-1],), f"Shape of g is {g.shape} should be {particles.shape[0]}"
        assert correction.shape == particles.shape, f"Shape of correction is {correction.shape} should be {particles.shape}"
        return np.einsum("j,ij->ij", g, grad) - correction, new_state

    return Preconditioner(init_fn, update_fn)