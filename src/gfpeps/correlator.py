import jax.numpy as jnp
from jax.scipy.linalg import inv
from .Gin import BatchGammaIn

def make_correlator(Lx=100, Ly=100, *, Nf, Nv):
    Gin = BatchGammaIn(Lx, Ly, Nv)
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for i in range(Nf + 4*Nv)])

    @jax.jit
    def correlator(T):
        # compuate energything from T
        Glocal = jnp.transpose(T) @ j_matrix @ T
        A, B, D = Glocal[0:Df, 0:Df], Glocal[0:Df, Df:], Glocal[Df:, Df:]
        return A + B @ inv(D+Gin) @ jnp.transpose(B) # for newer jax version.

    return correlator
