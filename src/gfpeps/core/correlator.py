import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import inv, block_diag
from .Gin import batched_Gin

def make_correlator(Lx=100, Ly=100, *, Nf, Nv):
    """
    Create a correlator function for a Gaussian fermionic Projected Entangled Pair State (gfPEPS).

    This function generates a JIT-compiled correlator that computes the correlation matrix
    for a given tensor T in the gfPEPS ansatz.

    Args:
        Lx (int): Number of lattice sites in the x-direction. Default is 100.
        Ly (int): Number of lattice sites in the y-direction. Default is 100.
        Nf (int): Number of physical fermion flavors per site.
        Nv (int): Number of virtual modes per bond.

    Returns:
        function: A JIT-compiled correlator function that takes a tensor T as input
                  and returns the correlation matrix.
    """
    Df = 2*Nf
    Gin = batched_Gin(Lx, Ly, Nv)
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for _ in range(Nf + 4*Nv)])

    @jit
    def correlator(T):
        """
        Compute the correlation matrix for a given tensor T.

        Args:
            T (jnp.ndarray): The tensor representing the gfPEPS ansatz.

        Returns:
            jnp.ndarray: The computed correlation matrix.
        """
        Glocal = jnp.transpose(T) @ j_matrix @ T
        A, B, D = Glocal[:Df, :Df], Glocal[:Df, Df:], Glocal[Df:, Df:]
        return A + B @ inv(D+Gin) @ jnp.transpose(B)

    return correlator
