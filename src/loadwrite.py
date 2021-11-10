import numpy as np
import jax.numpy as jnp
import h5py,os
from ABD import unitarize

def initialT(loadfile,Tsize):
    if loadfile != None and os.path.isfile(loadfile):
        print('Try to initialize T from',loadfile)
        with h5py.File(loadfile, 'r') as f:
            if "/transformer/T" in f.keys():
                T = f["/transformer/T"][:]
                f.close()
                return jnp.reshape(T,(Tsize,Tsize))
            else:
                print("Load Failed! No /transformer/T in",loadfile," switch to random initialize!")
                f.close()
                return jnp.array(np.random.rand(Tsize,Tsize))
    return jnp.array(np.random.rand(Tsize,Tsize))

def savelog(writefile,results):
    if writefile != None:
        print('Save T to',writefile)
        with h5py.File(writefile, 'w') as f:
            f["/transformer/T"] = results.x
            f["/energy/EABD"] = results.fun
            f["/optimize/EABD"] = results.fun
            f["/optimize/success"] = results.success
            f["/optimize/iterations"] = results.nit
            f["/optimize/normg"] = np.linalg.norm(results.jac)
            f.close()

def savelog_trivial(writefile,x,fun):
    if writefile != None:
        print('Save T to',writefile)
        with h5py.File(writefile, 'w') as f:
            f["/transformer/T"] = x
            f["/energy/EABD"] = fun
            f.close()