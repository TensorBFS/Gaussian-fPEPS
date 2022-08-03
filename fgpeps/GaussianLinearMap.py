import jax.numpy as jnp
from jax.scipy.linalg import inv
from ABD import getABD

def GaussianLinearMap(Glocal,Gin):
    r""" Glocal: Local project, Gin, Gamma in"""
    A,B,D = getABD(Glocal)
    return A + B @ inv(D+Gin) @ jnp.transpose(B) # for newer jax version.