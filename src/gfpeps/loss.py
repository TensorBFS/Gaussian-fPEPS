from jax import vmap
import jax.numpy as jnp
from .ABD import getGammaProjector
from .Gin import BatchGammaIn, BatchK
from .GaussianLinearMap import GaussianLinearMap
from .measure import compute_correlators

def energy_expectation(G, F, E_k, D_k):
    r"""Compute energy expectation value for given correlators and energy kernel.
    
    ⟨H⟩ = ∑_k Tr[E(k)G(k) + D(k)F(k) + D(k)^†F(k)^†]
    
    Args:
        G (jnp.ndarray): Normal Green's functions, shape (N_k, Nf, Nf)
        F (jnp.ndarray): Anomalous Green's functions, shape (N_k, Nf, Nf)
        E_k (jnp.ndarray): Normal energy kernel, shape (N_k, Nf, Nf)
        D_k (jnp.ndarray): Anomalous energy kernel, shape (N_k, Nf, Nf)
        
    Returns:
        float: Energy expectation value
    """
    # Compute energy contributions
    E_normal = jnp.einsum('kij,kji->k', E_k, G)
    E_pairing = jnp.einsum('kij,kji->k', D_k, F) + jnp.einsum('kij,kji->k', D_k.conj(), F.conj())
    
    return jnp.mean(E_normal + E_pairing)

def make_loss(kernel_fn, Lx=100, Ly=100, Nv=2, **kernel_params):
    r"""Create a loss function for variational optimization.
    
    Args:
        kernel_fn (callable): Function that generates energy kernels E(k), D(k)
        Lx, Ly (int): Number of k points
        Nv (int): Number of virtual modes
        **kernel_params: Parameters passed to kernel_fn
        
    Returns:
        callable: Loss function that takes tensor T as input
    """
    # Prepare k-points and energy kernels
    batch_k = BatchK(Lx, Ly)
    E_k, D_k = vmap(lambda k: kernel_fn(k, **kernel_params))(batch_k)
    BatchGin = BatchGammaIn(Lx, Ly, Nv)

    def lossT(T):
        r"""Compute energy expectation value for given tensor parameters.
        
        Args:
            T: Variational tensor parameters
            
        Returns:
            float: Real part of energy expectation value
        """
        Glocal = getGammaProjector(T, Nv)
        BatchGout = GaussianLinearMap(Glocal, BatchGin)
        G, F = compute_correlators(BatchGout)
        return jnp.real(energy_expectation(G, F, E_k, D_k))
    
    return lossT

# For backward compatibility
def optimize_runtime_loss(Lx=100, Ly=100, Nv=2, t=1.0, Dx=0.0, Dy=0.0, mu=0.0):
    r"""Legacy wrapper for Hubbard-BCS model optimization.
    
    Args:
        Lx, Ly (int): Number of k points
        Nv (int): Number of virtual modes
        t (float): Hopping amplitude
        Dx, Dy (float): Pairing potentials
        mu (float): Chemical potential
        
    Returns:
        callable: Loss function that takes tensor T as input
    """
    from .kernels import hubbard_bcs_kernel
    return make_loss(hubbard_bcs_kernel, Lx=Lx, Ly=Ly, Nv=Nv, 
                    t=t, Dx=Dx, Dy=Dy, mu=mu)