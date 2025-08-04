"""
Kitaev Honeycomb Spin-1/2 Model using Gaussian fPEPS

All-in-one implementation for maximum simplicity and clarity.
Physics: Kitaev honeycomb model with spin-1/2 Majorana representation.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.linalg import inv, block_diag
import pymanopt
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.optimizers import ConjugateGradient, SteepestDescent
from pymanopt import Problem
import logging
from dataclasses import dataclass
from jax.numpy.linalg import solve

# Fixed parameters for Kitaev spin-1/2 model
NF = 1  # Number of physical fermion flavors (fixed for Kitaev)

# ============================================================================
# Core Physics: Kitaev Honeycomb Kernel
# ============================================================================

def kitaev_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    """Kitaev honeycomb spin-1/2 energy kernel in Majorana basis.
    
    Args:
        k: Momentum vector (kx, ky)
        Jx, Jy, Jz: Kitaev coupling strengths
        
    Returns:
        2x2 antisymmetric matrix h(k) in Majorana basis
    """
    kx, ky = k[0], k[1]
    # J(k) = Jz - Jx*exp(ikx) - Jy*exp(iky) 
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    # Anti-Hermitian matrix
    
    # Jk = Jk / jnp.abs(Jk) # What if we Only take care the argument of Jk

    return jnp.array([[0, Jk], [-Jk.conj(), 0]]) / 4.0 # sigma_a -> S_a

# ============================================================================
# Momentum Space and Virtual Bonds
# ============================================================================

def batched_k(Lx, Ly):
    """Generate momentum points with APBC-PBC boundary conditions."""
    X, Y = jnp.meshgrid((jnp.arange(Lx))/Lx, jnp.arange(Ly)/Ly)
    return 2 * jnp.pi * jnp.array([X.flatten(), Y.flatten()]).T

def batched_Gin(Lx, Ly, Nv):
    """Generate virtual bond matrices Gamma_in for all momentum points."""
    def _gamma_in(k, Nv):
        def single_gamma(ki):
            t = jnp.exp(1j*ki)
            ct = -jnp.exp(-1j*ki)
            base = jnp.array([[0,0,0,t],[0,0,t,0],[0,ct,0,0],[ct,0,0,0]])
            return block_diag(*[base for _ in range(Nv)])
        return block_diag(*[single_gamma(ki) for ki in k])
    
    return vmap(lambda k: _gamma_in(k, Nv), 0)(batched_k(Lx, Ly))

# ============================================================================
# Gaussian Mapping and Correlator
# ============================================================================

def make_correlator(Lx, Ly, Nv):
    """Create correlator function specialized for Kitaev (Nf=1)."""
    Df = 2 * NF  # 2 for Kitaev
    Gin = batched_Gin(Lx, Ly, Nv)
    # Symplectic structure for Nf=1 + 4*Nv virtual modes
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for _ in range(NF + 4*Nv)])

    @jit
    def correlator(T):
        """Gaussian mapping: Γ_out = A + B(D + Γ_in)^(-1)B^T"""
        Glocal = jnp.transpose(T) @ j_matrix @ T
        A, B, D = Glocal[:Df, :Df], Glocal[:Df, Df:], Glocal[Df:, Df:]
        return A + B @ solve(D + Gin, B.T)
        # return A + B @ inv(D + Gin) @ jnp.transpose(B)

    return correlator

# ============================================================================
# Energy Loss Function
# ============================================================================

def make_loss(Lx, Ly, Nv, Jx=1.0, Jy=1.0, Jz=1.0):
    """Create energy loss function for Kitaev model."""
    # Prepare momentum points and energy kernels
    batch_k = batched_k(Lx, Ly)
    batch_h = vmap(lambda k: kitaev_kernel(k, Jx=Jx, Jy=Jy, Jz=Jz))(batch_k)
    correlator = make_correlator(Lx, Ly, Nv)

    @jit
    def loss_fn(T):
        """Compute energy expectation: ⟨H⟩ = Tr(Γ * h)"""
        Gout = correlator(T)
        return jnp.mean(jnp.real(Gout * batch_h)) * 2.0 # mean [0,1] and [1,0]

    return loss_fn

# ============================================================================
# Main Kitaev fPEPS Class
# ============================================================================

@dataclass
class KitaevfPEPS:
    """Simplified Kitaev fPEPS with fixed Nf=1."""
    Lx: int
    Ly: int  
    Nv: int
    Jx: float = 1.0
    Jy: float = 1.0
    Jz: float = 1.0
    seed: int = 42
    T: jnp.ndarray = None
    loss: callable = None

    def __post_init__(self):
        """Initialize random T matrix and loss function."""
        if self.T is None:
            self.T = self._initialize_random_T(seed=self.seed)
        self.loss = make_loss(self.Lx, self.Ly, self.Nv, 
                             Jx=self.Jx, Jy=self.Jy, Jz=self.Jz)

    def _initialize_random_T(self, seed=42):
        """Initialize random orthogonal matrix T."""
        dim = 2 * NF + 8 * self.Nv  # 2 + 8*Nv for Kitaev
        key = jax.random.PRNGKey(seed)
        T = jax.random.normal(key, shape=(dim, dim))
        U, _, V = jnp.linalg.svd(T)
        return U @ V

    def energy(self):
        """Compute current energy."""
        return self.loss(self.T)

# ============================================================================
# Optimization
# ============================================================================

def optimize(kitaev_system, max_iterations=1000, verbosity=1):
    """Optimize Kitaev fPEPS using Riemannian optimization."""
    dimT = kitaev_system.T.shape[0]
    
    # Optimization on Stiefel manifold (orthogonal matrices)
    manifold = Stiefel(dimT, dimT)
    
    @pymanopt.function.jax(manifold)
    def cost(x):
        return kitaev_system.loss(x)

    problem = Problem(manifold=manifold, cost=cost)
    solver = ConjugateGradient(
    # solver = SteepestDescent(
        verbosity=1, 
        min_gradient_norm=1e-08,
        min_step_size=1e-12,
        log_verbosity=verbosity, 
        max_iterations=max_iterations
    )

    result = solver.run(problem, initial_point=kitaev_system.T)
    kitaev_system.T = result.point
    return kitaev_system, result

# ============================================================================
# Configuration System
# ============================================================================

# Default configuration for Kitaev simulations
DEFAULT_CONFIG = {
    # System parameters
    "Lx": 51,                  # Number of k-points in x direction
    "Ly": 51,                  # Number of k-points in y direction  
    "Nv": 2,                   # Number of virtual modes per bond
    
    # Physics parameters
    "Jx": 1.0,                 # X-direction Kitaev coupling
    "Jy": 1.0,                 # Y-direction Kitaev coupling
    "Jz": 1.0,                 # Z-direction Kitaev coupling
    
    # Optimization parameters
    "max_iterations": 1000,    # Maximum optimization iterations
    "seed": 42,                # Random seed for initialization
    "verbosity": 1,            # Optimization verbosity level
    
    # Output parameters
    "workingdir": "./results", # Working directory for outputs
    "save_results": True,      # Whether to save optimization results
    "log_level": "INFO",       # Logging level
}

# ============================================================================
# High-level Interface
# ============================================================================

def run_kitaev_simulation(config=None):
    """
    Complete Kitaev simulation pipeline with flexible configuration.
    
    Args:
        config: Optional configuration dictionary. If None, use defaults.
                Can override any parameters in DEFAULT_CONFIG.
    
    Returns:
        tuple: (optimized_kitaev_system, optimization_result)
    
    Example:
        # Use defaults
        kitaev, result = run_kitaev_simulation()
        
        # Override specific parameters
        config = {"Lx": 100, "Ly": 100, "Jx": 0.5, "max_iterations": 500}
        kitaev, result = run_kitaev_simulation(config)
    """
    # Merge provided config with defaults
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        full_config = DEFAULT_CONFIG.copy()
        full_config.update(config)
        config = full_config
    
    # Enable double precision
    jax.config.update("jax_enable_x64", True)
    
    # Setup logging
    log_level = getattr(logging, config["log_level"].upper())
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Extract parameters
    Lx, Ly, Nv = config["Lx"], config["Ly"], config["Nv"]
    Jx, Jy, Jz = config["Jx"], config["Jy"], config["Jz"]
    max_iterations = config["max_iterations"]
    seed = config["seed"]
    verbosity = config["verbosity"]
    
    # Create Kitaev system
    logging.info(f"Initializing Kitaev system: {Lx}x{Ly}, Nv={Nv}")
    logging.info(f"Kitaev parameters: Jx={Jx}, Jy={Jy}, Jz={Jz}")
    
    kitaev = KitaevfPEPS(Lx=Lx, Ly=Ly, Nv=Nv, Jx=Jx, Jy=Jy, Jz=Jz, seed=seed)
    
    initial_energy = kitaev.energy()
    logging.info(f"Initial energy: {initial_energy:.8f}")
    
    # Optimize
    logging.info("Starting optimization...")
    optimized_kitaev, result = optimize(kitaev, max_iterations=max_iterations, 
                                       verbosity=verbosity)
    
    final_energy = optimized_kitaev.energy()
    logging.info(f"Final energy: {final_energy:.8f}")
    logging.info(f"Energy improvement: {initial_energy - final_energy:.8f}")
    
    # Save results if requested
    if config["save_results"]:
        _save_simulation_results(optimized_kitaev, result, config)
    
    return optimized_kitaev, result

def _save_simulation_results(kitaev_system, result, config):
    """Save simulation results including Glocal matrix."""
    import json
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    
    # Create Nv{N}seed{M} directory under results
    base_dir = Path(config["workingdir"])
    seed = config["seed"]
    Nv = config["Nv"]
    seed_dir = base_dir / f"Nv{Nv}seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp (no need for seed in filename since it's in directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kitaev_Jx{config['Jx']}_Jy{config['Jy']}_Jz{config['Jz']}_{timestamp}"
    
    # Compute Glocal = T^T @ J @ T (critical for future calculations)
    Nf = 1  # Fixed for Kitaev
    Nv = config["Nv"]
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for _ in range(Nf + 4*Nv)])
    Glocal = jnp.transpose(kitaev_system.T) @ j_matrix @ kitaev_system.T
    
    # Save T matrix and Glocal as numpy arrays
    T_matrix = np.array(kitaev_system.T)
    Glocal_matrix = np.array(Glocal)
    
    # Save matrices in seed directory
    np.save(seed_dir / f"{filename}_Glocal.npy", Glocal_matrix)
    
    # Prepare metadata
    results = {
        "config": config,
        "final_energy": float(kitaev_system.energy()),
        "iterations": len(result.log['iterations']['cost']),
        "converged": result.stopping_criterion,
        "loss_history": [float(x) for x in result.log['iterations']['cost']],
        "gradient_norms": [float(x) for x in result.log['iterations']['gradient_norm']],
        "timestamp": datetime.now().isoformat(),
        "T_shape": T_matrix.shape,
        "Glocal_shape": Glocal_matrix.shape,
        "seed_directory": str(seed_dir),
        "files": {
            "Glocal_matrix": f"{filename}_Glocal.npy"
        }
    }
    
    # Save metadata as JSON in seed directory
    with open(seed_dir / f"{filename}_meta.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to seed directory:")
    logging.info(f"  Directory: {seed_dir}")
    logging.info(f"  Glocal matrix: {seed_dir}/{filename}_Glocal.npy") 
    logging.info(f"  Metadata: {seed_dir}/{filename}_meta.json")

# ============================================================================
# Convenience Functions  
# ============================================================================

def quick_run(**kwargs):
    """Quick run with parameter overrides. Convenient for interactive use."""
    return run_kitaev_simulation(kwargs if kwargs else None)

def debug_run():
    """Run with minimal parameters for debugging."""
    debug_config = {
        "Lx": 5, "Ly": 5, "Nv": 1, 
        "max_iterations": 10,
        "log_level": "DEBUG"
    }
    return run_kitaev_simulation(debug_config) 