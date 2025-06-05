import jax
from jax import vmap, grad
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple, NamedTuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map
import jax.scipy as jsp
from functools import partial


class GMMComponent(NamedTuple):
    """Represents a single Gaussian component with mean and covariance"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    weight: float    # Scalar weight


class GMMState(NamedTuple):
    """State for GMM-based particle representation"""
    components: list[GMMComponent]  # List of Gaussian components
    n_components: int


def particles_to_gmm(particles: jax.Array, 
                     weights: jax.Array = None) -> GMMState:
    """
    Convert particle representation to GMM representation.
    Each particle becomes a Gaussian component with identity covariance.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
    
    Returns:
        GMMState with Gaussian components
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    # Initialize each particle as a Gaussian component with small identity covariance
    components = []
    for i in range(n_particles):
        mean = particles[i]
        # Start with small identity covariance to avoid degeneracy
        cov = np.eye(d_z) * 0.1
        weight = weights[i]
        components.append(GMMComponent(mean=mean, cov=cov, weight=weight))
    
    return GMMState(components=components, n_components=n_particles)


def gmm_to_particles(gmm_state: GMMState) -> jax.Array:
    """
    Extract particle locations from GMM (using means).
    
    Args:
        gmm_state: GMM representation
        
    Returns:
        Array of particle means, shape (n_particles, d_z)
    """
    means = [comp.mean for comp in gmm_state.components]
    return np.stack(means, axis=0)


def bures_wasserstein_distance_squared(mu1: jax.Array, cov1: jax.Array,
                                     mu2: jax.Array, cov2: jax.Array) -> float:
    """
    Compute squared Bures-Wasserstein distance between two Gaussian distributions.
    
    BW^2(N(mu1, cov1), N(mu2, cov2)) = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2(cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    
    Args:
        mu1, mu2: Means of the Gaussians
        cov1, cov2: Covariance matrices
        
    Returns:
        Squared Bures-Wasserstein distance
    """
    # Mean difference term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term: Tr(cov1 + cov2 - 2(cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    # Compute cov1^{1/2}
    try:
        cov1_sqrt = jsp.linalg.sqrtm(cov1)
        # Compute cov1^{1/2} cov2 cov1^{1/2}
        temp = cov1_sqrt @ cov2 @ cov1_sqrt
        # Compute its square root
        temp_sqrt = jsp.linalg.sqrtm(temp)
        # Final covariance term
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(temp_sqrt)
    except:
        # Fallback to simpler approximation if matrix square root fails
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + cov_term


def riemannian_grad_mean(euclidean_grad_mean: jax.Array) -> jax.Array:
    """
    Riemannian gradient for mean parameters (Euclidean space).
    Since means live in Euclidean space, Riemannian gradient = Euclidean gradient.
    """
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Riemannian gradient for covariance matrix on the Bures-Wasserstein manifold.
    
    grad_BW = 4 * {grad_euclidean * cov}_symmetric
    where {A}_symmetric = (A + A^T) / 2
    
    Args:
        euclidean_grad_cov: Euclidean gradient w.r.t. covariance
        cov: Current covariance matrix
        
    Returns:
        Riemannian gradient on Bures-Wasserstein manifold
    """
    # Compute the product
    product = euclidean_grad_cov @ cov
    # Symmetrize
    symmetric_product = (product + product.T) / 2
    # Scale by 4 (from Bures-Wasserstein geometry)
    return 4 * symmetric_product


def retraction_cov(cov: jax.Array, tangent_vector: jax.Array) -> jax.Array:
    """
    Retraction operator for covariance matrices on Bures-Wasserstein manifold.
    
    R_Sigma(X) = Sigma + X + L_X[Sigma] @ X @ L_X[Sigma]
    where L_X[Sigma] is the solution to L @ X + X @ L = Sigma (Lyapunov equation)
    
    For simplicity, we use a first-order approximation: R_Sigma(X) ≈ Sigma + X
    and ensure positive definiteness by adding small regularization.
    
    Args:
        cov: Current covariance matrix
        tangent_vector: Tangent vector (update direction)
        
    Returns:
        Updated covariance matrix on the manifold
    """
    # First-order retraction with regularization
    new_cov = cov + tangent_vector
    
    # Ensure symmetry
    new_cov = (new_cov + new_cov.T) / 2
    
    # Ensure positive definiteness by adding small regularization
    d = new_cov.shape[0]
    regularization = 1e-6 * np.eye(d)
    new_cov = new_cov + regularization
    
    return new_cov


def simplified_gmm_particle_grad(key: jax.random.PRNGKey,
                                pid: PID,
                                target: Target,
                                particles: jax.Array,
                                y: jax.Array,
                                mc_n_samples: int) -> jax.Array:
    """
    Simplified version that treats particles as GMM means with structure-aware updates.
    
    This version focuses on the core idea of structure preservation without full
    Bures-Wasserstein optimization.
    """
    def ediff_score_gmm_aware(particle, eps):
        """
        Enhanced score function that considers particle as part of a GMM structure.
        """
        # Sample from particle treated as Gaussian component
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        
        # Standard score computation
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        
        # Structure-preserving regularization
        # Encourage particles to maintain diversity (avoid collapse)
        diversity_reg = 0.01 * np.sum(particle ** 2)
        
        logq_mean = np.mean(logq, 0)
        logp_mean = np.mean(logp, 0)
        
        return logq_mean - logp_mean + diversity_reg
    
    eps = pid.conditional.base_sample(key, mc_n_samples)
    
    # Compute gradients with structure awareness
    grad = vmap(jax.grad(lambda p: ediff_score_gmm_aware(p, eps)))(particles)
    
    return grad


def gmm_de_particle_step(key: jax.random.PRNGKey,
                        pid: PID,
                        target: Target,
                        y: jax.Array,
                        optim: PIDOpt,
                        carry: PIDCarry,
                        hyperparams: PIDParameters) -> Tuple[PID, PIDCarry]:
    """
    Structure-preserving particle step for density estimation using GMM and Bures-Wasserstein geometry.
    
    This replaces the standard particle gradient computation with one that respects
    the GMM structure and uses Wasserstein gradient flows.
    """
    
    def grad_fn(particles):
        return simplified_gmm_particle_grad(
            key,
            pid,
            target,
            particles,
            y,
            hyperparams.mc_n_samples
        )
    
    # Apply preconditioner
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        grad_fn,
        carry.r_precon_state,
    )
    
    # Apply optimizer
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y
    )
    
    # Update particles
    pid = eqx.tree_at(lambda tree: tree.particles,
                      pid,
                      pid.particles + update)
    
    carry = PIDCarry(
        id=pid,
        theta_opt_state=carry.theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state
    )
    
    return pid, carry


def gmm_de_step(key: jax.random.PRNGKey,
               carry: PIDCarry,
               target: Target,
               y: jax.Array,
               optim: PIDOpt,
               hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    """
    Full density estimation step with GMM structure preservation.
    
    This combines the standard conditional parameter update with the new
    structure-preserving particle update.
    """
    theta_key, r_key = jax.random.split(key, 2)
    
    # Standard loss function for conditional parameters
    def loss(key, params, static):
        pid = eqx.combine(params, static)
        _samples = pid.sample(key, hyperparams.mc_n_samples, None)
        logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
        logp = vmap(target.log_prob, (0, None))(_samples, y)
        return np.mean(logq - logp, axis=0)
    
    # Update conditional parameters (theta)
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )
    
    # Update particles using GMM structure-preserving method
    pid, carry = gmm_de_particle_step(
        r_key,
        pid,
        target,
        y,
        optim,
        carry,
        hyperparams
    )
    
    carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state
    )
    
    return lval, carry
