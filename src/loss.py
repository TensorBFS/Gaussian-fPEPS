# define loss function
# Contains measuring

from functools import partial
from jax import vmap
import jax.numpy as jnp
from ABD import unitarize,getGammaProjector
from Gin import BatchGammaIn,BatchK
from GaussianLinearMap import GaussianLinearMap

def energy_function(hoping=1.0,DeltaX=0.0,DeltaY=0.0,Mu=0.0,Lx=100,Ly=100):
    batch_k = BatchK(Lx,Ly)
    batch_cosk = jnp.sum(jnp.cos(batch_k),axis=1)
    batch_delta = vmap(lambda x : jnp.vdot(x,jnp.array([DeltaX,DeltaY])))(jnp.cos(batch_k))

    def energy(BatchGout):
        r""" 
    only receive BatchΓout to calculate energy"""
        rhoup = 0.5 + 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,0:4:2,0:4:2],jnp.array([[0,-1.0],[1.0,0]]))
        rhodn = 0.5 + 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,1:4:2,1:4:2],jnp.array([[0,-1.0],[1.0,0]]))
        rho = rhoup+rhodn
        kappa = 0.25 * jnp.einsum('ijk,jk->i',BatchGout[:,0:4:2,1:4:2],jnp.array([[0,1.0],[1.0,0]]))
        return jnp.mean(-2*hoping * rho * batch_cosk + 4*batch_delta*kappa + Mu*rho)
    return energy

def optimize_runtime_loss(Lx=100,Ly=100,Nv=2,hoping=1.0,DeltaX=0.0,DeltaY=0.0,Mu=0.0):
    BatchGin = BatchGammaIn(Lx,Ly,Nv)
    energy = energy_function(hoping=hoping,DeltaX=DeltaX,DeltaY=DeltaY,Mu=Mu,Lx=Lx,Ly=Ly)
    # Eg = eg(BatchK(Lx,Ly),hoping,DeltaX,DeltaY,Mu)
    def lossT(T):
        r""" Maybe transform it to lossT will be helpful"""
        Glocal = getGammaProjector(T,Nv)
        BatchGout = GaussianLinearMap(Glocal,BatchGin)
        return jnp.real(energy(BatchGout))
    return lossT