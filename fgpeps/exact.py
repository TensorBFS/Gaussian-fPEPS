import jax.numpy as jnp
from jax.scipy.linalg import eigh
import jax
from Gin import BatchK

def exact(k,ht,D1X,D1Y,Mu):
    D = D1X*jnp.cos(k[0])+D1Y*jnp.cos(k[1])
    c = (jnp.cos(k[0])+jnp.cos(k[1]))
    t = jnp.array([[-ht*c,0],[0,-ht*c+Mu]])
    d = jnp.array([[0,D],[-D,0]])
    M = jnp.block([[t,d],[-d,-t]])
    w,v = eigh(M)
    N = w.shape[0]//2
    return (jnp.sum(w[0:N])+jnp.sum(jnp.diag(t)))

KSet = jnp.array([[1,0],[1,2],[1,3]])
def eg(Lx,Ly,ht,D1X,D1Y,Mu):
    KSet = BatchK(Lx,Ly)
    def f(k): return exact(k,ht,D1X,D1Y,Mu)
    return jnp.sum(jax.vmap(f)(KSet))/ KSet.shape[0]