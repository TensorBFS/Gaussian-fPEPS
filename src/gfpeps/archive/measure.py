from jax import vmap
import jax.numpy as jnp
from .Gin import BatchGammaIn
from .GaussianLinearMap import GaussianLinearMap

def compute_correlators(BatchGout, Nf=2):
    r"""Compute momentum-space fermionic correlators from Gaussian state.
    
    For each momentum k, computes the single-particle Green's functions:
    G_{αβ}(k) = ⟨c^†_{kα}c_{kβ}⟩ + 1/2 δ_{αβ}
    F_{αβ}(k) = ⟨c_{kα}c_{-kβ}⟩
    
    Args:
        BatchGout (jnp.ndarray): Batch of Gaussian correlation matrices for each k
        Nf (int, optional): Number of physical fermion flavors. Defaults to 2.
    
    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Returns:
            - G(k): Normal Green's functions, shape (N_k, Nf, Nf)
            - F(k): Anomalous Green's functions, shape (N_k, Nf, Nf)
            where N_k is the number of k points
    """
    N_k = BatchGout.shape[0]
    
    # Reshape BatchGout to separate fermion flavors
    # From (N_k, 2*Nf, 2*Nf) to (N_k, Nf, 2, Nf, 2)
    Gamma_reshaped = BatchGout.reshape(N_k, Nf, 2, Nf, 2)
    
    # Compute normal Green's functions
    # G_{ij}(k) = 1/2 δ_{ij} + 1/4 Tr[Γ_{kij}σ_y]
    G = (0.5 * jnp.eye(Nf)[None,:,:] + 
         0.25 * jnp.einsum('kabcd,dc->kab', 
                          Gamma_reshaped, 
                          jnp.array([[0,-1.0],[1.0,0]])))
    
    # Compute anomalous Green's functions
    # F_{ij}(k) = 1/4 Tr[Γ_{kij}σ_x]
    F = 0.25 * jnp.einsum('kabcd,dc->kab',
                         Gamma_reshaped,
                         jnp.array([[0,1.0],[1.0,0]]))
    
    return G, F
