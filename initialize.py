#!/usr/bin/env python3
"""
Local initialization around Dirac points for Kitaev honeycomb spin-1/2 fPEPS.

This module performs targeted optimization in a small momentum region around 
Dirac points to achieve better alignment of Gamma vortices with Dirac points.

Usage:
    python initialize.py                        # Use defaults
    python initialize.py --radius 0.5 --Nv 3   # Override parameters
    python optimize.py --initialize             # Auto-integrate with optimize.py
"""

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import inv, block_diag
import argparse
import logging
from pathlib import Path
import numpy as np

# Import from kitaev module
from kitaev import (
    NF, kitaev_kernel, KitaevfPEPS, optimize, 
    DEFAULT_CONFIG, _save_simulation_results
)

# Import for auto-analysis
try:
    from dirac_dense_analysis import analyze_dirac_scaling_dense
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("⚠️ dirac_dense_analysis.py not available - auto-analysis disabled")

# ============================================================================
# Local Momentum Space around Dirac Points
# ============================================================================

def batched_k_local(dirac_point, radius=0.2, L=21, offset=0.0):
    """
    Generate momentum points in a small rectangular region around Dirac point.
    
    Args:
        dirac_point: (kx, ky) coordinates of Dirac point center
        radius: Half-width of the rectangular region
        L: Number of k-points in each direction (Lx=Ly=L)
        offset: Small offset to avoid sampling exactly at Dirac point (in units of radius)
        
    Returns:
        k_points: Array of shape (L*L, 2) containing k-points
    """
    # Create local grid centered at Dirac point with small offset
    kx_center, ky_center = dirac_point
    
    # Add small offset to avoid sampling exactly at Dirac point
    offset_x = offset * radius * 0.1  # 10% of radius as offset
    offset_y = offset * radius * 0.1
    
    # Generate local grid with uniform spacing
    kx_local = jnp.linspace(kx_center - radius + offset_x, kx_center + radius + offset_x, L)
    ky_local = jnp.linspace(ky_center - radius + offset_y, ky_center + radius + offset_y, L)
    
    X, Y = jnp.meshgrid(kx_local, ky_local)
    return jnp.array([X.flatten(), Y.flatten()]).T

def batched_Gin_local(dirac_point, radius, L, Nv, offset=0.0):
    """Generate virtual bond matrices for local momentum region."""
    def _gamma_in(k, Nv):
        def single_gamma(ki):
            t = jnp.exp(1j*ki)
            ct = -jnp.exp(-1j*ki)
            base = jnp.array([[0,0,0,t],[0,0,t,0],[0,ct,0,0],[ct,0,0,0]])
            return block_diag(*[base for _ in range(Nv)])
        return block_diag(*[single_gamma(ki) for ki in k])
    
    return vmap(lambda k: _gamma_in(k, Nv), 0)(batched_k_local(dirac_point, radius, L, offset))

# ============================================================================
# Local Correlator and Loss Functions
# ============================================================================

def make_correlator_local(dirac_point, radius, L, Nv, offset=0.0):
    """Create correlator function for local momentum region."""
    Df = 2 * NF  # 2 for Kitaev
    Gin = batched_Gin_local(dirac_point, radius, L, Nv, offset)
    # Symplectic structure
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for _ in range(NF + 4*Nv)])

    @jax.jit
    def correlator(T):
        """Gaussian mapping: Γ_out = A + B(D + Γ_in)^(-1)B^T"""
        Glocal = jnp.transpose(T) @ j_matrix @ T
        A, B, D = Glocal[:Df, :Df], Glocal[:Df, Df:], Glocal[Df:, Df:]
        return A + B @ inv(D + Gin) @ jnp.transpose(B)

    return correlator

def make_loss_local(dirac_point, radius, L, Nv, Jx=1.0, Jy=1.0, Jz=1.0, offset=0.0):
    """Create energy loss function for local momentum region."""
    # Prepare local momentum points and energy kernels
    batch_k = batched_k_local(dirac_point, radius, L, offset)
    batch_h = vmap(lambda k: kitaev_kernel(k, Jx=Jx, Jy=Jy, Jz=Jz))(batch_k)
    correlator = make_correlator_local(dirac_point, radius, L, Nv, offset)

    @jax.jit
    def loss_fn(T):
        """Compute energy expectation for local region."""
        Gout = correlator(T)
        return jnp.sum(jnp.real(Gout * batch_h)) * 2.0

    return loss_fn

# ============================================================================
# Local Kitaev fPEPS Class
# ============================================================================

class KitaevfPEPSLocal(KitaevfPEPS):
    """Kitaev fPEPS optimized for local region around Dirac point."""
    
    def __init__(self, L, Nv, dirac_point=(jnp.pi/3, -jnp.pi/3), 
                 radius=0.2, Jx=1.0, Jy=1.0, Jz=1.0, seed=42, T=None, offset=0.0):
        # Don't call parent __init__ since we need custom loss function
        self.L = L
        self.Nv = Nv
        self.dirac_point = dirac_point
        self.radius = radius
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.seed = seed
        self.offset = offset
        
        if T is None:
            self.T = self._initialize_random_T(seed=seed)
        else:
            self.T = T
            
        # Create local loss function
        self.loss = make_loss_local(dirac_point, radius, L, Nv, Jx, Jy, Jz, offset)

# ============================================================================
# Local Initialization Pipeline
# ============================================================================

def run_local_initialization(config=None):
    """
    Run local initialization around Dirac point.
    
    Args:
        config: Configuration dictionary. Additional keys:
                - dirac_point: (kx, ky) coordinates of Dirac point
                - radius: Half-width of local region
                - local_Lx, local_Ly: Local grid size
                
    Returns:
        tuple: (locally_optimized_system, optimization_result)
    """
    # Default local configuration
    local_config = {
        "dirac_point": (jnp.pi/3, -jnp.pi/3),  # Known Dirac point for isotropic Kitaev
        "radius": 0.2,                          # Local region size
        "local_L": 101,                          # Local grid points (Lx=Ly=L)
        "offset": 0.1,                           # Small offset to avoid sampling exactly at Dirac point
        "max_iterations": 1000,                  # Fewer iterations for local optimization
        "log_level": "INFO",
        "save_results": True,                  # Don't save intermediate results
    }
    
    # Merge with provided config
    if config is None:
        config = local_config.copy()
    else:
        full_config = local_config.copy()
        full_config.update(config)
        config = full_config
    
    # Enable double precision
    jax.config.update("jax_enable_x64", True)
    
    # Setup logging
    log_level = getattr(logging, config["log_level"].upper())
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Extract parameters
    dirac_point = config["dirac_point"]
    radius = config["radius"]
    local_L = config["local_L"]
    offset = config.get("offset", 0.1)
    Nv = config.get("Nv", 2)
    Jx, Jy, Jz = config.get("Jx", 1.0), config.get("Jy", 1.0), config.get("Jz", 1.0)
    max_iterations = config["max_iterations"]
    seed = config.get("seed", 42)
    verbosity = config.get("verbosity", 1)
    
    # Create local Kitaev system
    logging.info(f"🎯 Initializing LOCAL Kitaev system around Dirac point:")
    logging.info(f"   Dirac point: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})")
    logging.info(f"   Local region: radius = {radius}")
    logging.info(f"   Local grid: {local_L}x{local_L} k-points")
    logging.info(f"   Offset: {offset} (to avoid sampling exactly at Dirac point)")
    logging.info(f"   Parameters: Nv={Nv}, Jx={Jx}, Jy={Jy}, Jz={Jz}")
    
    kitaev_local = KitaevfPEPSLocal(
        L=local_L, Nv=Nv,
        dirac_point=dirac_point, radius=radius,
        Jx=Jx, Jy=Jy, Jz=Jz, seed=seed, offset=offset
    )
    
    initial_energy = kitaev_local.energy()
    logging.info(f"Initial local energy: {initial_energy:.8f}")
    
    # Optimize locally
    logging.info("🔄 Starting LOCAL optimization...")
    optimized_local, result = optimize(kitaev_local, max_iterations=max_iterations, 
                                      verbosity=verbosity)
    
    final_energy = optimized_local.energy()
    logging.info(f"Final local energy: {final_energy:.8f}")
    logging.info(f"Local energy improvement: {initial_energy - final_energy:.8f}")
    
    return optimized_local, result

def create_global_system_from_local(local_system, global_config=None):
    """
    Create a global Kitaev system using T matrix from local optimization.
    
    Args:
        local_system: Locally optimized KitaevfPEPSLocal system
        global_config: Configuration for global system
        
    Returns:
        KitaevfPEPS: Global system initialized with local T matrix
    """
    if global_config is None:
        global_config = DEFAULT_CONFIG.copy()
    
    # Create global system with same T matrix from local optimization
    global_system = KitaevfPEPS(
        Lx=global_config["Lx"],
        Ly=global_config["Ly"], 
        Nv=local_system.Nv,
        Jx=local_system.Jx,
        Jy=local_system.Jy,
        Jz=local_system.Jz,
        seed=local_system.seed,
        T=local_system.T  # Use optimized T from local system
    )
    
    logging.info(f"🌍 Created global system ({global_config['Lx']}x{global_config['Ly']}) with locally optimized T matrix")
    logging.info(f"Global energy with local T: {global_system.energy():.8f}")
    
    return global_system

# ============================================================================
# Auto-Analysis Integration
# ============================================================================

def run_auto_analysis(saved_files_info, dirac_point, analysis_radius=0.001, 
                     grid_size=32):
    """
    Automatically run Dirac dense analysis on saved results.
    
    Args:
        saved_files_info: Dictionary containing info about saved files
        dirac_point: Dirac point coordinates for analysis
        analysis_radius: Half-width of square region around Dirac point
        grid_size: Number of points in each direction (grid_size x grid_size)
    """
    if not ANALYSIS_AVAILABLE:
        print("⚠️ Auto-analysis skipped: dirac_dense_analysis.py not available")
        return
    
    # Find the Glocal file
    glocal_file = None
    results_dir = None
    
    if 'seed_directory' in saved_files_info and 'files' in saved_files_info:
        seed_dir = Path(saved_files_info['seed_directory'])
        if 'Glocal_matrix' in saved_files_info['files']:
            glocal_filename = saved_files_info['files']['Glocal_matrix']
            glocal_file = seed_dir / glocal_filename
            results_dir = seed_dir
    
    if glocal_file is None or not glocal_file.exists():
        print("⚠️ Auto-analysis skipped: Glocal file not found")
        return
    
    if analysis_radius is None:
        analysis_radius = saved_files_info['config']['radius']

    print(f"\n🔬 STARTING AUTO-ANALYSIS...")
    print(f"   Glocal file: {glocal_file}")
    print(f"   Results directory: {results_dir}")
    print(f"   Analysis region: ±{analysis_radius}")
    print(f"   Grid size: {grid_size}x{grid_size}")
    
    # Generate output filename in the same directory
    analysis_filename = glocal_file.stem.replace('_Glocal', '_dirac_analysis')
    analysis_path = results_dir / f"{analysis_filename}.png"
    
    try:
        # Run the analysis
        results = analyze_dirac_scaling_dense(
            glocal_file=str(glocal_file),
            dirac_point=dirac_point,
            radius=analysis_radius,
            grid_size=grid_size,
            save_path=str(analysis_path)
        )
        
        print(f"✅ AUTO-ANALYSIS COMPLETE:")
        print(f"   Analysis plots saved: {analysis_path}")
        print(f"   2D plot saved: {analysis_path.with_name(analysis_path.stem + '_2d' + analysis_path.suffix)}")
        print(f"   Numerical data saved: {analysis_path.with_suffix('.npy')}")
        print(f"   Total sampling points: {results['n_points']}")
        print(f"   Valid data points: {len(results['energies'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Auto-analysis failed: {str(e)}")
        return None

def save_and_get_info(system, result, config):
    """
    Save simulation results and return file information for auto-analysis.
    """
    # Import required modules
    import json
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    from jax.scipy.linalg import block_diag
    
    # Create Nv{N}seed{M} directory under results
    base_dir = Path(config["workingdir"])
    seed = config["seed"]
    Nv = config["Nv"]
    seed_dir = base_dir / f"Nv{Nv}seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kitaev_Jx{config['Jx']}_Jy{config['Jy']}_Jz{config['Jz']}_{timestamp}"
    
    # Compute Glocal = T^T @ J @ T
    Nf = 1  # Fixed for Kitaev
    Nv = config["Nv"]
    j_matrix = block_diag(*[jnp.array([[0,1.0],[-1,0]]) for _ in range(Nf + 4*Nv)])
    Glocal = jnp.transpose(system.T) @ j_matrix @ system.T
    
    # Save matrices
    T_matrix = np.array(system.T)
    Glocal_matrix = np.array(Glocal)
    
    # Save Glocal matrix
    glocal_file = seed_dir / f"{filename}_Glocal.npy"
    np.save(glocal_file, Glocal_matrix)
    
    # Prepare metadata
    results = {
        "config": config,
        "final_energy": float(system.energy()),
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
    
    # Save metadata
    meta_file = seed_dir / f"{filename}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to seed directory:")
    logging.info(f"  Directory: {seed_dir}")
    logging.info(f"  Glocal matrix: {glocal_file}") 
    logging.info(f"  Metadata: {meta_file}")
    
    return results

# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local Dirac point initialization for Kitaev fPEPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Local optimization parameters
    parser.add_argument("--dirac_x", type=float, default=jnp.pi/3,
                       help="Dirac point x coordinate")
    parser.add_argument("--dirac_y", type=float, default=-jnp.pi/3,
                       help="Dirac point y coordinate")
    parser.add_argument("--radius", type=float, default=0.05,
                       help="Half-width of local momentum region")
    parser.add_argument("--local_L", type=int, default=51,
                       help="Number of k-points in each direction (Lx=Ly=L)")
    parser.add_argument("--offset", type=float, default=0.01,
                       help="Small offset to avoid sampling exactly at Dirac point (in units of radius)")
    
    # Physics parameters
    parser.add_argument("--Nv", type=int, default=2,
                       help="Virtual bond dimension")
    parser.add_argument("--Jx", type=float, default=1.0,
                       help="X-direction Kitaev coupling")
    parser.add_argument("--Jy", type=float, default=1.0,
                       help="Y-direction Kitaev coupling")
    parser.add_argument("--Jz", type=float, default=1.0,
                       help="Z-direction Kitaev coupling")
    
    # Optimization parameters
    parser.add_argument("--max_iterations", type=int, default=1000,
                       help="Maximum optimization iterations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbosity", type=int, default=1,
                       help="Optimization verbosity")
    
    # Output parameters
    parser.add_argument("--save_results", action='store_true',
                       help="Save optimization results")
    parser.add_argument("--workingdir", type=str, default="./results",
                       help="Working directory for outputs")
    
    # Global optimization option
    parser.add_argument("--continue_global", action='store_true',
                       help="Continue with global optimization after local initialization")
    parser.add_argument("--global_Lx", type=int, default=51,
                       help="Global grid size in x direction")
    parser.add_argument("--global_Ly", type=int, default=51,
                       help="Global grid size in y direction")
    parser.add_argument("--global_iterations", type=int, default=1000,
                       help="Global optimization iterations")
    
    # Auto-analysis option
    parser.add_argument("--auto_analysis", action='store_true',
                       help="Automatically run Dirac dense analysis after optimization")
    parser.add_argument("--analysis_radius", type=float, default=None,
                       help="Half-width of square region around Dirac point for dense analysis")
    parser.add_argument("--grid_size", type=int, default=32,
                       help="Number of points in each direction for dense analysis")
    
    return parser.parse_args()

def main():
    """Main entry point for local initialization."""
    args = parse_args()
    
    print("🎯 LOCAL DIRAC INITIALIZATION")
    print("Devices:", jax.devices())
    
    # Prepare local configuration
    local_config = {
        "dirac_point": (args.dirac_x, args.dirac_y),
        "radius": args.radius,
        "local_L": args.local_L,
        "offset": args.offset,
        "Nv": args.Nv,
        "Jx": args.Jx,
        "Jy": args.Jy, 
        "Jz": args.Jz,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "verbosity": args.verbosity,
        "save_results": args.save_results,
        "workingdir": args.workingdir,
    }
    
    # Run local initialization
    local_system, local_result = run_local_initialization(local_config)
    
    print(f"\n🎯 LOCAL INITIALIZATION COMPLETE:")
    print(f"   Final local energy: {local_system.energy():.8f}")
    print(f"   Local iterations: {len(local_result.log['iterations']['cost'])}")
    
    # Continue with global optimization if requested
    if args.continue_global:
        print(f"\n🌍 CONTINUING WITH GLOBAL OPTIMIZATION...")
        
        global_config = {
            "Lx": args.global_Lx,
            "Ly": args.global_Ly,
            "max_iterations": args.global_iterations,
            "verbosity": args.verbosity,
            "save_results": args.save_results,
            "workingdir": args.workingdir,
        }
        
        # Create global system from local optimization
        global_system = create_global_system_from_local(local_system, global_config)
        
        # Run global optimization
        from kitaev import run_kitaev_simulation
        final_global_config = DEFAULT_CONFIG.copy()
        final_global_config.update(global_config)
        final_global_config.update({
            "Nv": local_system.Nv,
            "Jx": local_system.Jx,
            "Jy": local_system.Jy,
            "Jz": local_system.Jz,
            "seed": local_system.seed,
        })
        
        # Replace T matrix in global system
        global_system.T = local_system.T
        optimized_global, global_result = optimize(global_system, 
                                                  max_iterations=args.global_iterations,
                                                  verbosity=args.verbosity)
        
        print(f"\n🌍 GLOBAL OPTIMIZATION COMPLETE:")
        print(f"   Final global energy: {optimized_global.energy():.8f}")
        print(f"   Global iterations: {len(global_result.log['iterations']['cost'])}")
        
        # Save final results
        saved_info = None
        if args.save_results:
            saved_info = save_and_get_info(optimized_global, global_result, final_global_config)
        
        # Run auto-analysis if requested
        if args.auto_analysis and saved_info:
            run_auto_analysis(
                saved_files_info=saved_info,
                dirac_point=(args.dirac_x, args.dirac_y),
                analysis_radius=args.analysis_radius,
                grid_size=args.grid_size
            )
        elif args.auto_analysis and not saved_info:
            print("⚠️ Auto-analysis skipped: results not saved (use --save_results)")
        
        return optimized_global.energy()
    
    else:
        # Save local results if requested
        saved_info = None
        if args.save_results:
            saved_info = save_and_get_info(local_system, local_result, local_config)
        
        # Run auto-analysis if requested
        if args.auto_analysis and saved_info:
            run_auto_analysis(
                saved_files_info=saved_info,
                dirac_point=(args.dirac_x, args.dirac_y),
                analysis_radius=args.analysis_radius,
                grid_size=args.grid_size
            )
        elif args.auto_analysis and not saved_info:
            print("⚠️ Auto-analysis skipped: results not saved (use --save_results)")
        
        return local_system.energy()

if __name__ == '__main__':
    main() 