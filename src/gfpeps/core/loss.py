
from jax import vmap, jit
import jax.numpy as jnp
from .Gin import batched_Gin, batched_k
from .correlator import make_correlator

def make_loss(kernel_fn, Lx=100, Ly=100, *, Nf=1, Nv=1, **kernel_params):
    r"""Create a loss function for variational optimization of a Gaussian fermionic Projected Entangled Pair State (gfPEPS).

    This function generates a loss function that computes the energy expectation value
    for a given set of variational tensor parameters. The loss function can be used
    in optimization procedures to find the ground state of the system.

    Args:
        kernel_fn (callable): Function that generates energy kernels E(k) and D(k).
            Should take momentum k and additional kernel_params as arguments.
        Lx (int): Number of k-points in the x-direction. Default is 100.
        Ly (int): Number of k-points in the y-direction. Default is 100.
        Nf (int): Number of fermion flavors. Default is 1.
        Nv (int): Number of virtual modes. Default is 1.
        **kernel_params: Additional parameters to be passed to kernel_fn.

    Returns:
        callable: A loss function that takes a tensor T as input and returns
            the real part of the energy expectation value.
    """
    # Prepare k-points and energy kernels
    batch_k = batched_k(Lx, Ly)
    batch_h = vmap(lambda k: kernel_fn(k, **kernel_params))(batch_k)
    correlator = make_correlator(Lx=Lx, Ly=Ly, Nf=Nf, Nv=Nv)

    @jit
    def lossT(T):
        r"""Compute energy expectation value for given tensor parameters.

        Args:
            T (jnp.ndarray): Variational tensor parameters representing the gfPEPS ansatz.

        Returns:
            float: Real part of the energy expectation value.
        """
        Gout = correlator(T)
        return jnp.mean(jnp.real(vmap(jnp.dot)(Gout, batch_h)))

    return lossT
