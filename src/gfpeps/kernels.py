"""
Energy kernels for various Hamiltonians in momentum space using Majorana representation.

Each kernel function returns the Hamiltonian matrix in Majorana basis:
H = i/2 ∑_k c_k^T M(k) c_{-k}
where c_k are Majorana operators and M(k) is antisymmetric.
"""

import jax.numpy as jnp

def bcs_pairing_kernel(k, t=1.0, Dx=0.0, Dy=0.0, mu=0.0):
    r"""Construct energy kernel for BCS pairing Hamiltonian in Majorana basis.
    
    Original complex fermion Hamiltonian:
    H = ∑_{k,αβ} E_{αβ}(k) c^†_{kα}c_{kβ} + [Δ_{αβ}(k)c_{kα}c_{-kβ} + h.c.]
    
    In Majorana basis (c_{k,2n} = (a_k + a^†_{-k})/√2, c_{k,2n+1} = i(a^†_{-k} - a_k)/√2):
    H = i/2 ∑_k c_k^T M(k) c_{-k}
    
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        t (float): Hopping amplitude
        Dx, Dy (float): Pairing potentials in x,y directions
        mu (float): Chemical potential
        
    Returns:
        jnp.ndarray: Hamiltonian matrix M(k) in Majorana basis, shape (4, 4)
    """
    # Single-particle dispersion
    eps_k = -2 * t * jnp.sum(jnp.cos(k))
    # k-dependent pairing
    delta_k = 2 * (Dx * jnp.cos(k[0]) + Dy * jnp.cos(k[1]))
    
    # Convert to Majorana basis
    # For spin-up: c_↑ = (γ₁ + iγ₂)/√2
    # For spin-dn: c_↓ = (γ₃ + iγ₄)/√2
    M_k = jnp.zeros((4, 4), dtype=complex)
    
    # Normal part contribution
    M_k = M_k.at[0,1].set((eps_k - mu))
    M_k = M_k.at[2,3].set((eps_k - mu))
    
    # Pairing contribution
    M_k = M_k.at[0,3].set(delta_k)
    M_k = M_k.at[1,2].set(delta_k)
    
    # Make antisymmetric
    M_k = M_k - M_k.T
    
    return M_k

def kitaev_honeycomb_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    r"""Construct energy kernel for Kitaev honeycomb model in Majorana basis.
    
    Original spin Hamiltonian:
    H = -∑_d ∑_⟨jk⟩∈d Jd σⱼᵈσₖᵈ
    
    After Majorana representation σⱼᵈ = icⱼᵈcⱼ⁰:
    H_eff = i ∑_d ∑_⟨jk⟩∈d Jd uⱼₖ cⱼ⁰cₖ⁰
    
    In momentum space with uⱼₖ = 1:
    H = i/2 ∑_k c_k^T M(k) c_{-k}
    where M(k) = J(k) for sublattice coupling
    
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        Jx, Jy, Jz (float): Coupling strengths
        
    Returns:
        jnp.ndarray: Hamiltonian matrix M(k) in Majorana basis, shape (2, 2)
    """
    # Use distorted Bravais lattice (square) for computation
    kx, ky = k[0], k[1]
    
    # J(k) = Jz - Jx exp(ikx) - Jy exp(iky)
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    
    # Construct antisymmetric matrix
    M_k = jnp.array([[0, Jk], [-Jk, 0]])
    
    return M_k