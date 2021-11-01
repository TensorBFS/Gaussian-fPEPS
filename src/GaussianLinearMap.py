# Gaussian Linear Map
# From an Input T or R to Gamma Out

import jax
import jax.numpy as jnp
from Gin import GammaIn
from jax.scipy.linalg import inv
from ABD import getABDfromT,getABDfromR,unitarize,getGammaProjector,getABD

@jax.jit
def GaussianLinearMap(Glocal,Gin,Nv):
    r""" Glocal: Local project, Gin, Gamma in
    """
    A,B,D = getABD(Glocal,Nv)
    def ABDGamma(Gin):
        return A + B @ inv(D+Gin) @ jnp.transpose(B)
    vmapABD = jax.vmap(ABDGamma,0)
    return vmapABD(Gin)

def getGoutfromR(R,Nv):
    r""" Get Gamma Out from R; dispatch to getGoutfromT"""
    return getGoutfromT(unitarize(R),Nv)

def getGoutfromT(T,Gin,Nv):
    Glocal = getGammaProjector(T,Nv)
    return GaussianLinearMap(Glocal,Gin,Nv)
