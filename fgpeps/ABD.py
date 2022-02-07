# Contains Functions to construct ABD

from functools import partial
from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax.scipy.linalg import expm

def getGammaProjector(T,Nv):
    r"""Get Gamma local from orthogonal matrix T"""
    return GammaProjector(T,J(Nv),Nv)

def J(Nv):
    r""" direct sum of [0 1;-1 0]; 8Nv+4 comes from 4 directions-> 2 in each direction.
     4 from up and down physical fermions """
    return block_diag(*[jnp.array([[0,1.0],[-1,0]]) for i in range(4*Nv+2)])

def GammaProjector(T,J,Nv):
    r""" Despite a transpose, obtain Gamma Projector """
    return jnp.transpose(T) @ J @ T

def getABD(GammaP):
    r"""get A,B,D from slice GammaProjector"""
    A = GammaP[0:4,0:4]
    B = GammaP[0:4,4:]
    D = GammaP[4:,4:]
    return A,B,D

def unitarize(R):
    r""" Construct a orthogonal matrix from an arbitary matrix R"""
    return expm(R-jnp.transpose(R))