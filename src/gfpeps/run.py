"""
Main entry point for Gaussian fPEPS calculations.

This module provides a simple example of using the Gaussian fPEPS implementation
for ground state optimization of a BCS pairing model on a square lattice.
"""
import os
import h5py
import logging
import jax
import jax.numpy as jnp
import numpy as np
from .GfPEPS import GaussianfPEPS
from .core.optim import optim
jax.config.update("jax_enable_x64", True)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run(Lx, Ly, Nv, kernel_names, kernel_params, max_iterations=1000, seed=42, simdir=None):
    """Run GfPEPS optimization with given parameters.
    
    Args:
        Lx (int): Number of k-points in x direction
        Ly (int): Number of k-points in y direction
        Nv (int): Number of virtual modes
        kernel_names (str): Name of the physical model kernel
        kernel_params (dict): Parameters for the kernel function
        max_iterations (int): Maximum optimization iterations
        seed (int): Random seed for initialization
        simdir (str, optional): Directory to save simulation results. Defaults to ~/simruns/gfpeps
    """
    # Set up logging and precision
    setup_logging()

    # Set default simdir and expand user path
    if simdir is None:
        simdir = os.path.expanduser('~/simruns/gfpeps')
    else:
        simdir = os.path.expanduser(simdir)
    
    # Create directory if it doesn't exist
    os.makedirs(simdir, exist_ok=True)

    # Convert paraeters to a concise string
    keyname = "_".join([key+"_"+str(value) for key, value in kernel_params.items()])

    # Try to load from loadfile
    filekey = kernel_names + ":" + "_".join(str(i) for i in kernel_params)+ 'Nv_' + str(Nv)
    loadfile = simdir + '/' + filekey + '.h5'
    try :
        with h5py.File(loadfile, 'r') as f:
            logging.info(f'Try to initialize T from {loadfile}')
            T = f["T"][:]
    except:
        logging.info(f"Load Failed! No {loadfile} switch to random initialize!")
        T = None

    # Create GfPEPS instance
    logging.info("Initializing GfPEPS")
    gfpeps = GaussianfPEPS(
        Lx=Lx,
        Ly=Ly,
        Nv=Nv,
        kernel_names=kernel_names,
        kernel_params=kernel_params,
        seed=seed,
        T=T
    )
    
    # Run optimization
    logging.info("Starting optimization")
    optimized_gfpeps, result = optim(
        gfpeps,
        log_verbosity=1,
        max_iterations=max_iterations
    )
    
    # Generate final report
    logging.info("\nOptimization Results:")
    logging.info("-" * 50)
    logging.info(f"Final Energy: {optimized_gfpeps.loss(optimized_gfpeps.T):.8f}")
    logging.info(f"Model Parameters:")
    logging.info(f"  - Lattice size: {optimized_gfpeps.Lx}x{optimized_gfpeps.Ly}")
    logging.info(f"  - Virtual modes (Nv): {optimized_gfpeps.Nv}")
    logging.info(f"  - Physical fermions (Nf): {optimized_gfpeps.Nf}")
    logging.info(f"  - Kernel: {optimized_gfpeps.kernel_names}")
    for param, value in optimized_gfpeps.kernel_params.items():
        logging.info(f"  - {param}: {value}")
    logging.info("-" * 50)
    
    # save all information in results
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])
    
    logging.info("Iterations \t Cost \t Gradient Norm")
    for iter in range(len(log_cost)):
        logging.info(f"{iter} \t {log_cost[iter]} \t {log_gnorm[iter]}")

    logging.info(f"Optimization done!, final cost: {result.cost}, gnorm: {result.gradient_norm }")

    # Save results to file        
    savefile = simdir + '/' + filekey + '.h5'
    logging.info(f'Save T to {savefile}')
    with h5py.File(savefile, 'w') as f:
        f["T"] = optimized_gfpeps.T
        for iter in range(len(log_cost)):
            f[f"point/{iter}"] = result.log["iterations"]["point"][iter]
            f[f"cost/{iter}"] = log_cost[iter]
            f[f"gnorm/{iter}"] = log_gnorm[iter]

    return 0