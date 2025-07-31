#!/usr/bin/env python3
"""
Analysis module for Kitaev honeycomb fPEPS results.

This module reads saved Glocal matrices and performs:
1. Band dispersion calculations
2. Real-space correlation function G(r) analysis
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import inv, block_diag

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Fixed parameters for Kitaev
NF = 1  # Number of physical fermion flavors (fixed for Kitaev)

# ============================================================================
# Core Functions from kitaev.py (needed for analysis)
# ============================================================================

def batched_k(Lx, Ly):
    """Generate momentum points in [-π, π] range to match exact_dispersion.py."""
    X, Y = jnp.meshgrid((jnp.arange(Lx)+0.5)/Lx, jnp.arange(Ly)/Ly)
    # Transform from [0,1] to [-π, π] range
    kx = 2 * jnp.pi * (X - 0.5)  # Center at 0, range [-π, π]
    ky = 2 * jnp.pi * (Y - 0.5)  # Center at 0, range [-π, π]
    return jnp.array([kx.flatten(), ky.flatten()]).T

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
# Analysis Functions
# ============================================================================

def compute_correlator_from_glocal(Glocal, Lx, Ly, Nv):
    """
    Compute correlator from saved Glocal matrix with new k-space sampling.
    
    Args:
        Glocal: Saved Glocal matrix (from optimization)
        Lx, Ly: New k-space sampling parameters
        Nv: Virtual bond dimension (must match the saved Glocal)
    
    Returns:
        correlator: k-space correlator matrices, shape (Lx*Ly, 2*NF, 2*NF)
        k_points: corresponding k-points
    """
    Df = 2 * NF  # 2 for Kitaev
    
    # Generate new Gin with desired k-space sampling
    Gin = batched_Gin(Lx, Ly, Nv)
    k_points = batched_k(Lx, Ly)
    
    # Extract A, B, D from Glocal
    A = Glocal[:Df, :Df]
    B = Glocal[:Df, Df:]
    D = Glocal[Df:, Df:]
    
    # Compute correlator for each k-point: Γ_out = A + B(D + Γ_in)^(-1)B^T
    def compute_single_correlator(gin):
        return A + B @ inv(D + gin) @ jnp.transpose(B)
    
    correlator = vmap(compute_single_correlator)(Gin)
    
    return correlator, k_points

def compute_band_dispersion(correlator, k_points, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Compute band dispersion E(k) from correlator and Hamiltonian kernel.
    
    For each k-point: E(k) = sum(correlator_k * H_k)
    This gives the energy expectation value for the fPEPS state.
    
    Args:
        correlator: k-space correlator matrices, shape (Lx*Ly, 2*NF, 2*NF)
        k_points: corresponding k-points, shape (Lx*Ly, 2)
        Jx, Jy, Jz: Kitaev coupling parameters
    
    Returns:
        eigenvalues: Band energies E(k), shape (Lx*Ly, 2)
    """
    from kitaev import kitaev_kernel
    
    def compute_energy_for_k(corr_k, k):
        # Get the 2x2 Hamiltonian matrix for this k-point
        H_k = kitaev_kernel(k, Jx, Jy, Jz)
        # Compute energy: E = mean(real(dot(correlator, Hamiltonian)))
        # This matches the implementation in kitaev.py
        dot_result = jnp.dot(corr_k, H_k)
        energy = jnp.mean(jnp.real(dot_result))
        # Ensure energy is real and return [E, -E] for the two bands
        energy_real = jnp.real(energy)
        return jnp.array([energy_real, -energy_real])
    
    # Vectorized computation over all k-points
    eigenvals = vmap(compute_energy_for_k)(correlator, k_points)
    
    return eigenvals

def compute_real_space_correlations(correlator, k_points, Lx, Ly, max_distance=10):
    """
    Compute real-space correlation function G(r) via Fourier transform, with optional zero-padding.
    Args:
        correlator: k-space correlator matrices (Kx*Ky, ...)
        k_points: corresponding k-points
        Lx, Ly: real-space grid size for zero-padding (can be > Kx, Ky)
        max_distance: maximum distance for G(r) calculation
    Returns:
        G_r: real-space correlations, shape (2*max_distance+1, 2*max_distance+1, ...)
        r_points: corresponding real-space points
    """
    Kx = int(jnp.sqrt(len(k_points)))
    Ky = Kx
    correlator_2d = correlator.reshape(Kx, Ky, 2*NF, 2*NF)
    # Zero-pad to Lx, Ly
    pad_x = (Lx - Kx) // 2
    pad_y = (Ly - Ky) // 2
    pad_width = ((pad_x, Lx-Kx-pad_x), (pad_y, Ly-Ky-pad_y), (0,0), (0,0))
    correlator_padded = jnp.pad(correlator_2d, pad_width)
    # Fourier transform to real space
    G_r_full = jnp.fft.ifft2(correlator_padded, axes=(0, 1))
    G_r_full = jnp.fft.fftshift(G_r_full, axes=(0, 1))
    # Extract central region around r=0
    center_x, center_y = Lx // 2, Ly // 2
    x_slice = slice(center_x - max_distance, center_x + max_distance + 1)
    y_slice = slice(center_y - max_distance, center_y + max_distance + 1)
    G_r = G_r_full[x_slice, y_slice]
    # Generate real-space coordinate grid centered at (0,0)
    r_x = jnp.arange(-max_distance, max_distance + 1)
    r_y = jnp.arange(-max_distance, max_distance + 1)
    r_points = jnp.stack(jnp.meshgrid(r_x, r_y, indexing='ij'), axis=-1)
    return G_r, r_points

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_band_dispersion(eigenvalues, k_points, save_path=None):
    """Plot high-quality band dispersion using PRL-level plotting."""
    try:
        from plot.dispersion import plot_kitaev_bands
        
        fig, ax = plot_kitaev_bands(
            eigenvalues=np.array(eigenvalues), 
            k_points=np.array(k_points),
            save_path=save_path
        )
        plt.close(fig)
        return fig, ax
        
    except ImportError:
        # Fallback to basic plotting if plot module not available
        print("⚠️  High-quality plot module not available, using basic plotting")
        plt.figure(figsize=(10, 6))
        
        # For simplicity, plot along kx direction (ky=0)
        Lx = int(jnp.sqrt(len(k_points)))
        
        # Extract kx=0 to π line (ky=0)
        kx_line_indices = jnp.arange(0, Lx)
        kx_values = k_points[kx_line_indices, 0]
        eigenvals_line = eigenvalues[kx_line_indices]
        
        # Plot all bands
        for band in range(eigenvals_line.shape[1]):
            plt.plot(kx_values, eigenvals_line[:, band], 'o-', markersize=3, linewidth=1.5, 
                    label=f'Band {band+1}')
        
        plt.xlabel('kx')
        plt.ylabel('Energy')
        plt.title('Band Dispersion (ky = 0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Band dispersion plot saved: {Path(save_path).name}")
        
        plt.close()

def plot_real_space_correlations(G_r, r_points, component=(0,0), save_path=None):
    """Plot high-quality real-space correlation function using PRL-level plotting."""
    try:
        from plot.correlation import plot_kitaev_correlation
        
        fig, ax = plot_kitaev_correlation(
            G_r=np.array(G_r), 
            r_points=np.array(r_points),
            component=component,
            save_path=save_path
        )
        plt.close(fig)
        return fig, ax
        
    except ImportError:
        # Fallback to basic plotting if plot module not available
        print("⚠️  High-quality plot module not available, using basic plotting")
        plt.figure(figsize=(8, 6))
        
        # Plot magnitude of correlations
        G_magnitude = jnp.abs(G_r[:, :, component[0], component[1]])
        
        plt.imshow(G_magnitude, origin='lower', cmap='RdBu_r', 
                   extent=[r_points[0,0,0], r_points[0,-1,0], r_points[0,0,1], r_points[-1,0,1]])
        plt.colorbar(label=f'|G(r)| component ({component[0]},{component[1]})')
        plt.xlabel('rx')
        plt.ylabel('ry')
        plt.title(f'Real-space Correlation Function |G(r)| - Component ({component[0]},{component[1]})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Real-space correlations plot saved: {Path(save_path).name}")
        
        plt.close()

# ============================================================================
# File I/O Functions
# ============================================================================

def load_glocal_from_file(filepath):
    """Load Glocal matrix and metadata from saved files."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        # Direct .npy file
        Glocal = np.load(filepath)
        
        # Try to load corresponding metadata
        meta_file = filepath.with_name(filepath.stem.replace('_Glocal', '_meta') + '.json')
        metadata = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
    
    elif filepath.suffix == '.json':
        # Metadata file
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # Load Glocal from corresponding .npy file
        glocal_file = filepath.with_name(filepath.stem.replace('_meta', '_Glocal') + '.npy')
        if glocal_file.exists():
            Glocal = np.load(glocal_file)
        else:
            raise FileNotFoundError(f"Glocal file not found: {glocal_file}")
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return jnp.array(Glocal), metadata

def save_analysis_results(results, seed_dir, prefix="analysis"):
    """Save analysis results to seed directory."""
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results
    for key, value in results.items():
        if isinstance(value, jnp.ndarray):
            np.save(seed_dir / f"{prefix}_{key}.npy", np.array(value))
    
    # Save metadata
    metadata = {k: v for k, v in results.items() if not isinstance(v, jnp.ndarray)}
    with open(seed_dir / f"{prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Analysis results saved to seed directory: {seed_dir}")

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_analysis(glocal_file, Kx=100, Ky=100, Lx=100, Ly=100, max_distance=10, save_plots=True, output_dir=None):
    """
    Complete analysis pipeline with separate k-space and real-space grid sizes.
    Args:
        glocal_file: Path to Glocal file (.npy or .json)
        Kx, Ky: k-space sampling for analysis
        Lx, Ly: real-space grid size for zero-padding (can be > Kx, Ky)
        max_distance: maximum distance for G(r) calculation
        save_plots: whether to save plots
        output_dir: optional override for output directory (default: auto-detect seed dir)
    """
    print(f"🔬 Starting analysis of: {glocal_file}")
    # Load Glocal and metadata
    Glocal, metadata = load_glocal_from_file(glocal_file)
    # Extract original parameters
    if metadata and 'config' in metadata:
        original_config = metadata['config']
        Nv = original_config['Nv']
        original_Lx = original_config['Lx']
        original_Ly = original_config['Ly']
        seed = original_config['seed']
        print(f"📋 Original parameters: Lx={original_Lx}, Ly={original_Ly}, Nv={Nv}, seed={seed}")
        print(f"📋 Analysis parameters: Kx={Kx}, Ky={Ky}, Lx={Lx}, Ly={Ly}, max_distance={max_distance}")
    else:
        raise ValueError("Could not extract Nv from metadata. Please specify manually.")
    # Determine output directory - use Nv{N}seed{M} directory by default
    if output_dir is None:
        glocal_path = Path(glocal_file)
        parent_name = glocal_path.parent.name
        if parent_name.startswith('Nv') and 'seed' in parent_name:
            seed_dir = glocal_path.parent
        elif parent_name.startswith('seed'):
            results_dir = glocal_path.parent.parent
            Nv = original_config['Nv']
            seed_dir = results_dir / f"Nv{Nv}seed{seed}"
            seed_dir.mkdir(exist_ok=True)
        elif 'seed_directory' in metadata:
            seed_dir = Path(metadata['seed_directory'])
        else:
            results_dir = glocal_path.parent
            Nv = original_config['Nv']
            seed_dir = results_dir / f"Nv{Nv}seed{seed}"
            seed_dir.mkdir(exist_ok=True)
    else:
        seed_dir = Path(output_dir)
    print(f"📁 Saving analysis to: {seed_dir}")
    # Compute correlator with new k-space sampling
    print("🔄 Computing correlator...")
    correlator, k_points = compute_correlator_from_glocal(Glocal, Kx, Ky, Nv)
    # Compute TRUE band dispersion E(k) from Hamiltonian
    print("📊 Computing TRUE band dispersion E(k)...")
    Jx, Jy, Jz = original_config.get('Jx', 1.0), original_config.get('Jy', 1.0), original_config.get('Jz', 1.0)
    eigenvalues = compute_band_dispersion(correlator, k_points, Jx, Jy, Jz)
    # Compute real-space correlations
    print("🌐 Computing real-space correlations...")
    G_r, r_points = compute_real_space_correlations(correlator, k_points, Lx, Ly, max_distance)
    # Save numerical results
    results = {
        'correlator': correlator,
        'k_points': k_points,
        'eigenvalues': eigenvalues,
        'G_r': G_r,
        'r_points': r_points,
        'analysis_Lx': Lx,
        'analysis_Ly': Ly,
        'max_distance': max_distance,
        'original_config': metadata.get('config', {}),
        'seed': seed
    }
    
    save_analysis_results(results, seed_dir)
    
    # Generate plots in same seed directory
    if save_plots:
        print("🎨 Generating PRL-quality plots...")
        
        # Generate high-quality band dispersion plot (removed - using diagonal plot instead)
        
        # Generate high-quality correlation plots for all components
        n_components = G_r.shape[2]
        for i in range(n_components):
            for j in range(n_components):
                plot_real_space_correlations(
                    G_r, r_points, component=(i,j),
                    save_path=seed_dir / f"correlation_G{i}{j}_hq.png"
                )
        
        # Generate additional specialized plots if high-quality modules available
        try:
            from plot.dispersion import DispersionPlotter
            from plot.correlation import (CorrelationPlotter, plot_correlation_decay, 
                                        plot_correlation_scaling_analysis, 
                                        plot_correlation_scaling_analysis_diagonal)
            
            print("🎨 Generating additional specialized plots...")
            
            # Density of states
            disp_plotter = DispersionPlotter()
            disp_plotter.plot_density_of_states(
                eigenvalues=np.array(eigenvalues),
                save_path=seed_dir / "density_of_states.png"
            )
            
            # Enhanced correlation analysis for each component
            print("🔬 Performing power law scaling analysis...")
            n_components = G_r.shape[2]
            for i in range(n_components):
                for j in range(n_components):
                    # Power law scaling analysis (x-direction)
                    plot_correlation_scaling_analysis(
                        G_r=np.array(G_r), 
                        r_points=np.array(r_points),
                        component=(i,j),
                        direction='x',
                        save_path=seed_dir / f"correlation_scaling_G{i}{j}_x.png"
                    )
                    
                    # Power law scaling analysis (diagonal x+y direction)
                    plot_correlation_scaling_analysis_diagonal(
                        G_r=np.array(G_r), 
                        r_points=np.array(r_points),
                        component=(i,j),
                        save_path=seed_dir / f"correlation_scaling_G{i}{j}_diagonal.png"
                    )
                    
                    # Skip 4-fold copy - not needed with proper fftshift centering
            
            # Simplified band dispersion along diagonal line (0,2π) to (2π,0)
            from plot.dispersion import plot_diagonal_band_dispersion
            plot_diagonal_band_dispersion(
                eigenvalues=np.array(eigenvalues),
                k_points=np.array(k_points),
                save_path=seed_dir / "diagonal_band_dispersion.png"
            )
            
            # Dirac point scaling analysis with known Dirac point location
            from plot.dispersion import plot_dirac_scaling_analysis, plot_2d_band_structure_with_dirac_search
            print("🔬 Performing Dirac point scaling analysis...")
            
            # Create 2D band structure plot to visualize the band structure
            fig_2d, ax_2d, _ = plot_2d_band_structure_with_dirac_search(
                eigenvalues=np.array(eigenvalues),
                k_points=np.array(k_points),
                save_path=seed_dir / "2d_band_structure_with_dirac_search.png"
            )
            plt.close(fig_2d)
            
            # Use the known Dirac point location: (π/3, -π/3) in [-π, π] system
            known_dirac_point = (np.pi/3, -np.pi/3)  # ≈ (1.047, -1.047)
            print(f"🎯 Using known Dirac point: {known_dirac_point}")
            
            # Perform Dirac scaling analysis at the known Dirac point
            fig_dirac, axes_dirac = plot_dirac_scaling_analysis(
                eigenvalues=np.array(eigenvalues),
                k_points=np.array(k_points),
                dirac_point=known_dirac_point,
                radius=0.15,  # Small radius for dense sampling near Dirac point
                save_path=seed_dir / "dirac_scaling_analysis.png"
            )
            plt.close(fig_dirac)
            
            print(f"🔍 Analyzed Dirac point: {known_dirac_point}")
            
            print("✨ All specialized plots generated successfully!")
            
        except ImportError:
            print("ℹ️  Some specialized plotting features not available")
            print("   Install scipy>=1.7.0 and seaborn>=0.11.0 for full functionality")
    
    print("✅ Analysis complete!")
    return results

# ============================================================================
# Command Line Interface  
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Kitaev fPEPS results: band dispersion and real-space correlations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("glocal_file", type=str, 
                       help="Path to Glocal .npy file or metadata .json file")
    parser.add_argument("--Kx", type=int, default=100,
                       help="k-space sampling in x direction (for correlator)")
    parser.add_argument("--Ky", type=int, default=100, 
                       help="k-space sampling in y direction (for correlator)")
    parser.add_argument("--Lx", type=int, default=100,
                       help="real-space grid size in x direction (for zero-padding)")
    parser.add_argument("--Ly", type=int, default=100, 
                       help="real-space grid size in y direction (for zero-padding)")
    parser.add_argument("--max_distance", type=int, default=10,
                       help="Maximum distance for G(r) calculation")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-detect seed directory)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    results = run_analysis(
        glocal_file=args.glocal_file,
        Kx=args.Kx,
        Ky=args.Ky,
        Lx=args.Lx,
        Ly=args.Ly,
        max_distance=args.max_distance,
        save_plots=not args.no_plots,
        output_dir=args.output_dir
    )
    print(f"\n📈 Analysis Summary:")
    print(f"   Number of k-points: {len(results['k_points'])}")
    print(f"   Number of bands: {results['eigenvalues'].shape[1]}")
    print(f"   G(r) grid size: {results['G_r'].shape[:2]}")
    print(f"   Results saved to: {args.output_dir if args.output_dir else 'auto-detected seed directory'}")

if __name__ == '__main__':
    main() 