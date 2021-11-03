from args import args
import jax
from jax.config import config
from loadwrite import initialR   
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit,value_and_grad
from loss import optimize_runtime_loss
from loadwrite import initialR,savelog

if __name__ == '__main__':
# TEMP:
    Nv = args.Nv
    Rsize = 8*Nv+4
    LoadKey = "/home/qiyang/source/jaxgfpeps/result/Nv{}C{}.h5".format(args.Nv,args.loadlabel)
    Key = "/home/qiyang/source/jaxgfpeps/result/Nv{}C{}.h5".format(args.Nv,args.label)
#
    R = initialR(LoadKey,Rsize)
#
    lossR = jit(optimize_runtime_loss(Nv=args.Nv,Lx=args.Lx,Ly=args.Ly,
    hoping=args.ht,DeltaX=args.DeltaX,DeltaY=args.DeltaY),backend='gpu')
#
    # print(lossR(R))
    # dR  = jit(jax.grad(lossR))(R)
    # print(dR)
    # dR2 = jit(jax.hessian(lossR))(R)
    # print(dR2)
#
    import scipy.optimize
    def val_grad_R(R):
        value, grad = value_and_grad(lossR)(jnp.reshape(jnp.array(R),(Rsize,Rsize)))
        return np.array(value), np.array(grad).ravel()
#
    # results = scipy.optimize.minimize(val_grad_R, np.array(R).ravel(),
    #                                 method="L-BFGS-B", jac=True, options={'gtol': 1e-07,'maxiter': 1000,'disp':True})
    #
    def hessian_f(R):
        return np.reshape(np.array(jax.hessian(lossR)(jnp.reshape(R,(Rsize,Rsize)))),(Rsize*Rsize,Rsize*Rsize))
#
    results = scipy.optimize.minimize(val_grad_R, np.array(R).ravel(),
                                    method=args.optimizer, jac=True, hess=hessian_f ,options={'gtol': args.gtol,'maxiter': args.MaxIter,'disp': args.OptimDisp==1 })
    #
    print("success:", results.success, "\nniterations:", results.nit, "\nfinal loss:", results.fun)
    #
    savelog(Key,results)