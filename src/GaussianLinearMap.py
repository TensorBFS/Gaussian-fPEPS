# Gaussian Linear Map
# From an Input T or R to Gamma Out

import jax
import jax.numpy as jnp
from jax.scipy.linalg import inv
from ABD import getABD

def GaussianLinearMap(Glocal,Gin):
    r""" Glocal: Local project, Gin, Gamma in"""
    A,B,D = getABD(Glocal)
    def ABDGamma(Gin):
        return A + B @ inv(D+Gin) @ jnp.transpose(B)
    vmapABD = jax.vmap(ABDGamma,0)
    return vmapABD(Gin)