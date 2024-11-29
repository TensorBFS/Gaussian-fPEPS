"""
Energy kernels for various Hamiltonians in momentum space using Majorana representation.

Each kernel function returns the Hamiltonian matrix in Majorana basis:
H = i/2 ∑_k γ_k^T h(k) γ_{-k}
where γ_k are Majorana operators and h(k) is antisymmetric.
"""

import jax.numpy as jnp

def bcs_pairing_square_kernel(k, t=1.0, Dx=0.0, Dy=0.0, mu=0.0):
    r"""Construct energy kernel for BCS pairing Hamiltonian in Majorana basis.
    
    Original complex fermion Hamiltonian:
    H = const + ∑_{k,αβ} ε_{αβ}(k) (c^†_{kα}c_{kβ} - 1/2) + [Δ_{αβ}(k)c_{kα}c_{-kβ} + h.c.]
    where:
    - ε_{αβ}(k) = ε(k)δ_{αβ}
    - ε(k) = -2t(cos kx + cos ky) - μ
    - Δ_{αβ}(k) = Δ(k)δ_{α↑}δ_{β↓}
    - Δ(k) = 2(Dx cos kx + Dy cos ky)
    
    In Majorana basis:
    H = i/2 ∑_k γ_k^T h(k) γ_{-k}
    
    The matrix h(k) has the following structure:
    h(k) = 1/4 [
        [0, -ε(k), 0, Δ(k)],
        [ε(k), 0, Δ(k), 0],
        [0, -Δ(k), 0, -ε(k)],
        [-Δ(k), 0, ε(k), 0]
    ]
    
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        t (float): Hopping amplitude
        Dx, Dy (float): Pairing potentials in x,y directions
        mu (float): Chemical potential
        
    Returns:
        jnp.ndarray: Hamiltonian matrix h(k) in Majorana basis, shape (4, 4)
    """
    # Single-particle dispersion
    eps_k = -2.0 * t * jnp.sum(jnp.cos(k)) - mu
    # k-dependent pairing
    delta_k = 2.0 * (Dx * jnp.cos(k[0]) + Dy * jnp.cos(k[1]))
    
    # Construct 4×4 antisymmetric matrix in Majorana basis
    h_k = 0.25 * jnp.array([
        [0, -eps_k, 0, delta_k],
        [eps_k, 0, delta_k, 0],
        [0, -delta_k, 0, -eps_k],
        [-delta_k, 0, eps_k, 0]
    ])
    
    return h_k