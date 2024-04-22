# Exact Solve density of electrons in BCS state
import numpy as np
import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from scipy.optimize import root_scalar

Nv = 2
L = 101
KX,KY = jnp.meshgrid((jnp.arange(L)-0.5)/L,(jnp.arange(L))/L)
KXY = 2 * jnp.pi* jnp.array([KX.flatten(),KY.flatten()]).T

def nk(k,delta,mu):
    kx,ky = k
    a = 1/delta*(-jnp.cos(kx)-jnp.cos(ky)+mu/2)/jnp.abs(jnp.cos(kx)-jnp.cos(ky))
    return 1/jnp.sqrt(1+a**2)/(jnp.sqrt(1+a**2)+a)

def ntotal(delta,mu):
    nsumoverk = jax.vmap(lambda x: nk(x,delta,mu))
    return jnp.mean(nsumoverk(KXY))

def solve_mu(dxy,delta):
    def f(x): return ntotal(dxy,x)-(1-delta)
    return root_scalar(f,bracket=(0.0,10.0)).root
