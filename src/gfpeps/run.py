"""
Main entry point for Gaussian fPEPS calculations.

This module provides a simple example of using the Gaussian fPEPS implementation
for ground state optimization of a BCS pairing model on a square lattice.
"""

import logging
import jax
import jax.numpy as jnp
from .GfPEPS import GaussianfPEPS
from .core.optim import optim

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run(Lx, Ly, Nv, kernel_names, kernel_params, max_iterations=1000, seed=42):
    """Run GfPEPS optimization with given parameters.
    
    Args:
        Lx (int): Number of k-points in x direction
        Ly (int): Number of k-points in y direction
        Nv (int): Number of virtual modes
        kernel_names (str): Name of the physical model kernel
        kernel_params (dict): Parameters for the kernel function
        max_iterations (int): Maximum optimization iterations
        seed (int): Random seed for initialization
    """
    # Set up logging and precision
    setup_logging()
    jax.config.update("jax_enable_x64", True)
    
    # Create GfPEPS instance
    logging.info("Initializing GfPEPS")
    gfpeps = GaussianfPEPS(
        Lx=Lx,
        Ly=Ly,
        Nv=Nv,
        kernel_names=kernel_names,
        kernel_params=kernel_params,
        seed=seed
    )
    
    # Run optimization
    logging.info("Starting optimization")
    optimized_gfpeps = optim(
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
    
    return optimized_gfpeps