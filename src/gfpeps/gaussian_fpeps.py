import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit
from .loss import optimize_runtime_loss
from .loadwrite import initialT,savelog,savelog_trivial
from .exact import eg
from .deltatomu import solve_mu
import logging

# import Manopt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions,ConjugateGradient

def gaussian_fpeps(cfg):
    # unpack cfg
    np.random.seed(cfg.params.seed)
    Nv = cfg.params.Nv
    Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
    LoadKey, WriteKey  = cfg.file.LoadFile, cfg.file.WriteFile
    
    cfgh = cfg.hamiltonian
    ht = cfgh.ht
    DeltaX, DeltaY = cfgh.DeltaX, cfgh.DeltaY
    delta, Mu = cfgh.delta, cfgh.Mu

    Tsize = 8*Nv+4
    T = initialT(LoadKey,Tsize)
    U,S,V = np.linalg.svd(T)
    T = U @ V

    if cfgh.solve_mu_from_delta:
        logging.info("Overwrite Origin Mu")
        Mu = solve_mu(DeltaX,delta)
    
    lossT = jit(optimize_runtime_loss(Nv=Nv,Lx=Lx,Ly=Ly,
    hoping=ht,DeltaX=DeltaX,DeltaY=DeltaY,Mu=Mu), backend=cfg.backend)
    
    def egrad(x): return np.array(jax.grad(lossT)(jnp.array(x)))

    @jax.jit
    def hvp_loss(primals, tangents): return jax.jvp(jax.grad(lossT), primals, tangents)[1]

    def ehessa(x,v): return np.array(hvp_loss((jnp.array(x),), (jnp.array(v),)))

    Eg = eg(Lx,Ly,ht,DeltaX,DeltaY,Mu) # Will use solved Mu to calculate Eg
    logging.info("Eg = {}\n".format(Eg))
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


    problem = Problem(manifold=manifold, 
                      cost=cost,
                      euclidean_gradient=euclidean_gradient,
                      euclidean_hessian=euclidean_hessian)
    solver = ConjugateGradient(log_verbosity=1, max_iterations=cfg.optimizer.MaxIter)

    result = solver.run(problem, initial_point=T)
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])

    logging.info("Iterations \t Cost \t Gradient Norm")
    for iter in range(len(log_cost)):
        logging.info(f"{iter} \t {log_cost[iter]} \t {log_gnorm[iter]}")

    logging.info(f"Optimization done!, final cost: {result.cost}, gnorm: {result.gradient_norm }")
    
    Xopt = result.point
    args = {"Mu":Mu,"DeltaX":DeltaX,"DeltaY":DeltaY,"delta":delta,
            "ht":ht,"Lx":Lx,"Ly":Ly,"Nv":Nv,"seed":cfg.params.seed}
    savelog_trivial(WriteKey,Xopt,lossT(Xopt),Eg,args)
    return Xopt