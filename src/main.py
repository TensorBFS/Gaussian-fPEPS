import jax
from jax.config import config   
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit,value_and_grad
from loss import optimize_runtime_loss

lossR = jit(optimize_runtime_loss(Nv=2,Lx=100,Ly=100))

if __name__ == '__main__':
# TEMP:
    Nv = 2
    R = jnp.array(np.random.rand(20,20))
    print(lossR(R))
    dR  = jit(jax.grad(lossR))(R)
    # print(dR)
    # dR2 = jit(jax.hessian(lossR))(R)
    # print(dR2)

    import scipy.optimize
    def value_and_grad_numpy(f):
        def val_grad_f(R):
            value, grad = value_and_grad(f)(jnp.reshape(R,(20,20)))
            return np.array(value), np.array(grad).ravel()
        return val_grad_f
    results = scipy.optimize.minimize(value_and_grad_numpy(lossR), np.array(R).ravel(),
                                    method="L-BFGS-B", jac=True, options={'gtol': 1e-07,'maxiter': 1000,'disp':True})
    
    def hessian_f(R):
        return np.reshape(np.array(jax.hessian(lossR)(jnp.reshape(R,(20,20)))),(400,400))

    # results = scipy.optimize.minimize(value_and_grad_numpy(lossR), np.array(R).ravel(),
    #                                 method="trust-ncg", jac=True, hess=hessian_f ,options={'gtol': 1e-07,'maxiter': 50,'disp':True})
    
    print("success:", results.success, "\nniterations:", results.nit, "\nfinal loss:", results.fun)