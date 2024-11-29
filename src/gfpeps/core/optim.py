import logging
from ..GfPEPS import GaussianfPEPS
import pymanopt
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt import Problem
import numpy as np

def optim(gfpeps:GaussianfPEPS, log_verbosity=1, max_iterations=1000):
    dimT = gfpeps.T.shape[0]
    
    manifold = Stiefel(dimT, dimT)
    @pymanopt.function.jax(manifold)
    def cost(x):
        return gfpeps.loss(x)

    problem = Problem(manifold=manifold, 
                      cost=cost)
    solver = ConjugateGradient(log_verbosity=log_verbosity, max_iterations=max_iterations)

    result = solver.run(problem, initial_point=gfpeps.T)
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])
    
    logging.info("Iterations \t Cost \t Gradient Norm")
    for iter in range(len(log_cost)):
        logging.info(f"{iter} \t {log_cost[iter]} \t {log_gnorm[iter]}")

    logging.info(f"Optimization done!, final cost: {result.cost}, gnorm: {result.gradient_norm }")

    gfpeps.T = result.x
    return gfpeps