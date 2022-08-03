import jax
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit
from loss import optimize_runtime_loss
from loadwrite import initialT,savelog,savelog_trivial
from exact import eg
from deltatomu import solve_mu

# import Manopt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions,ConjugateGradient

def gaussian_fpeps(Nv=3,  # Number of virtual fermions on each bond
                Lx=101, # System size
                Ly=101, # System size
                ht=1.0, # Hoping term in BCS hamiltonian
                DeltaX= 1.0,
                DeltaY=-1.0,
                delta=0.0, # the density of holes, Has effect only if solve_mu is True
                Mu=0.0, # mu is the chemical potential, 
                solve_mu_from_delta = False, # If this is True, we will solve the chemical potential to match the density of electrons in BCS state
                LoadFile="./data/default.h5", # Load initial T from this file, if it exists
                WriteFile="./data/default.h5", # Write information about final Gaussian fPEPS to this file
                seed=42, # Random Number Generator seed
                MaxIter=100, # Maximum number of iterations
                gtol=1E-7, # gtol for optimizer
                backend="gpu"
                ):

    np.random.seed(seed)
    Nv = Nv
    Tsize = 8*Nv+4
    
    LoadKey = LoadFile
    Key = WriteFile
    T = initialT(LoadKey,Tsize)
    U,S,V = np.linalg.svd(T)
    T = U @ V

    if solve_mu_from_delta:
        print("Overwrite Origin Mu")
        Mu = solve_mu(DeltaX,delta)
    
    lossT = jit(optimize_runtime_loss(Nv=Nv,Lx=Lx,Ly=Ly,
    hoping=ht,DeltaX=DeltaX,DeltaY=DeltaY,Mu=Mu),backend=backend)
    
    def egrad(x): return np.array(jax.grad(lossT)(jnp.array(x)))

    @jax.jit
    def hvp_loss(primals, tangents): return jax.jvp(jax.grad(lossT), primals, tangents)[1]

    def ehessa(x,v): return np.array(hvp_loss((jnp.array(x),), (jnp.array(v),)))

    Eg = eg(Lx,Ly,ht,DeltaX,DeltaY,Mu) # Will use solved Mu to calculate Eg
    print("Eg = {}\n".format(Eg))
    # Optimizer

    manifold = Stiefel(Tsize, Tsize)
    @pymanopt.function.numpy(manifold)
    def cost(x):
        return lossT(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(x):
        return egrad(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(x,y):
        return ehessa(x,y)


    problem = Problem(manifold=manifold, cost=cost,euclidean_gradient=euclidean_gradient,euclidean_hessian=euclidean_hessian)
    solver = ConjugateGradient()

    result = solver.run(problem)
    Xopt = result.point
    args = {"Mu":Mu,"DeltaX":DeltaX,"DeltaY":DeltaY,"delta":delta,
            "ht":ht,"Lx":Lx,"Ly":Ly,"Nv":Nv,"seed":seed}
    savelog_trivial(Key,Xopt,lossT(Xopt),Eg,args)

if __name__ == '__main__':
    gaussian_fpeps()