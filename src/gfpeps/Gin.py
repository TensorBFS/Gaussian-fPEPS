from jax import vmap
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

def SingleGammaIn(k):
    r""" Create a single Gamma In"""
    t = jnp.exp(1j*k)
    ct = -jnp.exp(-1j*k)
    return jnp.array([[0,0,0,t],[0,0,t,0],[0,ct,0,0],[ct,0,0,0]])

def NvGammaIn(k,Nv):
    r""" Create a series Gamma In for k for Nv times -> Gamma In a direction"""
    temp = SingleGammaIn(k)
    return block_diag(*[temp for i in range(Nv)])

def GammaIn(k,Nv):
    r""" Create Gamma In for k for Nv times -> Gamma In all direction
    The order follow order in input k
    """
    return block_diag(*[NvGammaIn(ki,Nv) for ki in k])

def BatchK(Lx,Ly):
    X,Y = jnp.meshgrid((jnp.arange(Lx)-0.5)/Lx,(jnp.arange(Ly))/Ly)
    G = 2 * jnp.pi* jnp.array([X.flatten(),Y.flatten()]).T 
    return G

def BatchGammaIn(Lx,Ly,Nv):
    r"""Generate BZ for APBC-PBC boundary condition, return a batch of Lx,Ly
    """
    batch_k = BatchK(Lx,Ly)
    def GammaInNv(k):
        return GammaIn(k,Nv)
    return vmap(GammaInNv,0)(batch_k)
