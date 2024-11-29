"""
Energy kernels for various Hamiltonians in momentum space using Majorana representation.

Each kernel function returns the Hamiltonian matrix in Majorana basis:
H = i/2 ∑_k γ_k^T h(k) γ_{-k}
where γ_k are Majorana operators and h(k) is antisymmetric.
"""

import jax.numpy as jnp

def kitaev_honeycomb_1half_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    r"""Construct energy kernel for Kitaev honeycomb model in Majorana basis.
    
    Original spin Hamiltonian:
    H = -∑_d ∑_⟨jk⟩∈d Jd σⱼᵈσₖᵈ
    
    After Majorana representation σⱼᵈ = icⱼᵈcⱼ⁰:
    H_eff = i ∑_d ∑_⟨jk⟩∈d Jd uⱼₖ γⱼ⁰γₖ⁰
    where uⱼₖ = icⱼᵈcₖᵈ is the Z₂ gauge field
    
    In momentum space with uⱼₖ = 1:
    H = i/2 ∑_k γ_k^T h(k) γ_{-k}
    where h(k) = [
        [0, J(k)],
        [-J(k), 0]
    ]
    J(k) = Jz - Jx exp(ikx) - Jy exp(iky)
    
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        Jx, Jy, Jz (float): Coupling strengths
        
    Returns:
        jnp.ndarray: Hamiltonian matrix h(k) in Majorana basis, shape (2, 2)
    """
    # Use distorted Bravais lattice (square) for computation
    kx, ky = k[0], k[1]
    
    # J(k) = Jz - Jx exp(ikx) - Jy exp(iky)
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    
    # Construct antisymmetric matrix
    h_k = jnp.array([[0, Jk], [-Jk, 0]])
    
    return h_k