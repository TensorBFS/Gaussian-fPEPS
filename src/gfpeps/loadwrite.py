import numpy as np
import jax.numpy as jnp
import h5py,os
import logging

def initialT(loadfile,Tsize):
    if loadfile != None and os.path.isfile(loadfile):
        logging.info(f'Try to initialize T from {loadfile}')
        with h5py.File(loadfile, 'r') as f:
            if "/transformer/T" in f.keys():
                T = f["/transformer/T"][:]
                f.close()
                return jnp.reshape(T,(Tsize,Tsize))
            else:
                logging.debug(f"Load Failed! No /transformer/T in {loadfile} switch to random initialize!")
                f.close()
                return jnp.array(np.random.rand(Tsize,Tsize))
    return jnp.array(np.random.rand(Tsize,Tsize))

def savelog(writefile,results):
    if writefile != None:
        logging.info(f'Save T to {writefile}')
        with h5py.File(writefile, 'w') as f:
            f["/transformer/T"] = results.x
            f["/energy/EABD"] = results.fun
            f["/optimize/EABD"] = results.fun
            f["/optimize/success"] = results.success
            f["/optimize/iterations"] = results.nit
            f["/optimize/normg"] = np.linalg.norm(results.jac)
            f.close()

def savelog_trivial(writefile,x,fun,Eg,args,cor):
    if writefile != None:
        rhoup, rhodn, kappa = cor
        logging.info(f'Save T to {writefile}')
        with h5py.File(writefile, 'w') as f:
            f["/transformer/T"] = x

            f["/energy/EABD"] = fun
            f["/energy/Eg"] = Eg

            f["/model/Mu"] = args["Mu"]
            f["/model/DeltaX"] = args["DeltaX"]
            f["/model/DeltrY"] = args["DeltaY"]
            f["/model/Hoping"] = args["ht"]
            f["/model/Nv"] = args["Nv"]
            f["/model/seed"] = args["seed"]
            f["/model/Lx"] = args["Lx"]
            f["/model/Ly"] = args["Ly"]
            f["/model/delta"] = args["delta"]
            
            f["/cor/rhoup"] = rhoup
            f["/cor/rhodn"] = rhodn
            f["/cor/kappa"] = kappa
        
            f.close()