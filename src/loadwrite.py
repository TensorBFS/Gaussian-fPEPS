import numpy as np
import jax.numpy as jnp
import h5py,os
from ABD import unitarize

def initialR(loadfile,Rsize):
    if loadfile != None and os.path.isfile(loadfile):
        print('Try to initialize R from',loadfile)
        with h5py.File(loadfile, 'r') as f:
            if "/transformer/R" in f.keys():
                R = f["/transformer/R"][:]
                f.close()
                return jnp.reshape(R,(Rsize,Rsize))
            else:
                print("Load Failed! No /transformer/R in",loadfile," switch to random initialize!")
                f.close()
                return jnp.array(np.random.rand(Rsize,Rsize))
    return jnp.array(np.random.rand(Rsize,Rsize))

def savelog(writefile,results):
    if writefile != None:
        print('Save R,T to',writefile)
        with h5py.File(writefile, 'w') as f:
            f["/transformer/R"] = results.x
            f["/energy/EABD"] = results.fun
            f["/optimize/EABD"] = results.fun
            f["/optimize/success"] = results.success
            f["/optimize/iterations"] = results.nit
            f["/optimize/normg"] = np.linalg.norm(results.jac)
            f.close()


        
