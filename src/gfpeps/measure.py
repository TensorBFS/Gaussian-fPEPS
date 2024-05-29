from jax import vmap
import jax.numpy as jnp
from .ABD import getGammaProjector
from .Gin import BatchGammaIn,BatchK
from .GaussianLinearMap import GaussianLinearMap
import numpy as np

def measure(cfg, T):
    BatchGin = BatchGammaIn(cfg.lattice.Lx,cfg.lattice.Ly,cfg.params.Nv)
    Glocal = getGammaProjector(T,cfg.params.Nv)
    BatchGout = GaussianLinearMap(Glocal,BatchGin)
    rhoup = 0.5 + 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,0:4:2,0:4:2],jnp.array([[0,-1.0],[1.0,0]]))
    rhodn = 0.5 + 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,1:4:2,1:4:2],jnp.array([[0,-1.0],[1.0,0]]))
    kappa = 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,0:4:2,1:4:2],jnp.array([[0,1.0],[1.0,0]]))

    rhoup, rhodn, kappa = map(lambda x: np.reshape(x,(cfg.lattice.Lx,cfg.lattice.Ly)),(rhoup,rhodn,kappa))
    # BZ symmetry ?
    return rhoup, rhodn, kappa
