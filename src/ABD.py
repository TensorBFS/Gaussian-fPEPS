# Contains Functions to construct ABD

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
    return jnp.transpose(T) @ J @T

def getABD(GammaP):
    r"""get A,B,D from slice GammaProjector"""
    A = GammaP[0:4,0:4]
    B = GammaP[0:4,4:]
    D = GammaP[4:,4:]
    return A,B,D

def unitarize(R):
    r""" Construct a orthogonal matrix from an arbitary matrix R"""
    return expm(R-jnp.transpose(R))

# Not implemented and used ------------------------

def perm(Nv):
    r""" for simplicity , use perm getting from Julia fGPSP program, only for Nv = 1,2,3
    """
    if Nv == 1:
        return jnp.array([9, 7, 1, 2,11, 5,10, 8, 3, 4,12, 6])
    elif Nv == 2:
        return jnp.array([13,17, 7,11, 1, 2,15,19, 5, 9,14,18, 8,12, 3, 4,16,20, 6,10])
    elif Nv == 3:
        return jnp.array([17,21,25, 7,11,15, 1, 2,19,23,27, 5, 9,13,18,22,26, 8,12,16, 3, 4,20,24,28, 6,10,14])
    else:
        print("Warn: perm for Nv=",Nv,"is not implemented!\n")
        return None

def permute_matrix(G,order):
    r""" Maybe we can optimize ABD first and retract the original T by retrive this permutation"""
    return None