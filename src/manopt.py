from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import TrustRegions
# https://pymanopt.org/docs/api-reference.html#solvers
# https://pymanopt.org/docs/api-reference.html#module-pymanopt.core.problem


# (1) Instantiate a manifold
manifold = Stiefel(5, 2)

# (2) Define the cost function (here using autograd.numpy)
def cost(X): return np.sum(X)


problem = Problem(manifold=manifold, cost=cost, heissian=None)

# (3) Instantiate a Pymanopt solver
solver = TrustRegions()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)