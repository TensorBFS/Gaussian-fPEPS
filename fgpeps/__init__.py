import jax
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit
from loss import optimize_runtime_loss
from loadwrite import initialT
from exact import eg
from deltatomu import solve_mu

# Manopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import TrustRegions,ConjugateGradient

def fgpeps(args):
    np.random.seed(args.seed)
    Nv = args.Nv
    Tsize = 8*Nv+4

    T = initialT(None,Tsize)
    U,S,V = np.linalg.svd(T)
    T = U @ V

    Mu = solve_mu(args.DeltaX,args.delta)

    lossT = jit(optimize_runtime_loss(Nv=args.Nv,Lx=args.Lx,Ly=args.Ly,
    hoping=args.ht,DeltaX=args.DeltaX,DeltaY=args.DeltaY,Mu=Mu),backend='gpu')

    def egrad(x): return np.array(jax.grad(lossT)(jnp.array(x)))

    Eg = eg(args.Lx,args.Ly,args.ht,args.DeltaX,args.DeltaY,Mu) # Will use solved Mu to calculate Eg
    print("Eg = {}\n".format(Eg))

    manifold = Stiefel(Tsize, Tsize)
    problem = Problem(manifold=manifold, cost=lossT,egrad=egrad)
    solver = ConjugateGradient(maxiter=args.MaxIter)

    Xopt = solver.solve(problem,x=T)
    return Xopt

if __name__ == "__main__":
    from args import args
    fgpeps(args)