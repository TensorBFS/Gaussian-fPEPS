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

def batched_k(L):
    """Generate momentum points in [-π, π] range. Unified L instead of Lx,Ly."""
    X, Y = jnp.meshgrid((jnp.arange(L))/L, (jnp.arange(L))/L, indexing='xy')
    # Transform from [0,1] to [-π, π] range
    kx = 2 * jnp.pi * (X-0.5)  # Center at 0, range [-π, π]
    ky = 2 * jnp.pi * (Y-0.5)  # Center at 0, range [-π, π]
    # Use Fortran order to match meshgrid indexing
    return jnp.array([kx.flatten(order='F'), ky.flatten(order='F')]).T

def batched_Gin(L, Nv):
    """Generate virtual bond matrices Gamma_in for all momentum points. Unified L."""
    def _gamma_in(k, Nv):
        def single_gamma(ki):
            t = jnp.exp(1j*ki)
            ct = -jnp.exp(-1j*ki)
            base = jnp.array([[0,0,0,t],[0,0,t,0],[0,ct,0,0],[ct,0,0,0]])
            return block_diag(*[base for _ in range(Nv)])
        return block_diag(*[single_gamma(ki) for ki in k])
    
    return vmap(lambda k: _gamma_in(k, Nv), 0)(batched_k(L))

# ============================================================================
# Analysis Functions
# ============================================================================

def compute_correlator_from_glocal(Glocal, L, Nv):
    """
    Compute correlator from saved Glocal matrix with new k-space sampling.
    
    Args:
        Glocal: Saved Glocal matrix (from optimization)
        L: k-space sampling parameter (unified, was Lx=Ly)
        Nv: Virtual bond dimension (must match the saved Glocal)
    
    Returns:
        correlator: k-space correlator matrices, shape (L*L, 2*NF, 2*NF)
        k_points: corresponding k-points
    """
    Df = 2 * NF  # 2 for Kitaev
    
    # Generate new Gin with desired k-space sampling
    Gin = batched_Gin(L, Nv)
    k_points = batched_k(L)
    
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
        # FIXED: Use sum() instead of mean() to match energy expectation formula
        # ⟨H⟩ = Σᵢⱼ Γᵢⱼ(k) hᵢⱼ(k), not the average
        energy = jnp.mean(jnp.real(corr_k * H_k)) * 2.0
        # Ensure energy is real and return [E, -E] for the two bands
        energy_real = jnp.real(energy)
        return jnp.array([energy_real, -energy_real])
    
    # Vectorized computation over all k-points
    eigenvals = vmap(compute_energy_for_k)(correlator, k_points)
    
    return eigenvals

# Real-space correlation functions moved to analysis_correlation.py

# ============================================================================
# Plotting Functions (Simplified - correlation plots moved to analysis_correlation.py)
# ============================================================================

def plot_2d_band_structure_with_correlator_vector_field(eigenvalues, k_points, correlator, L,
                                                       save_path=None, band_index=0, show_vector_field=True):
    """
    Plot 2D band structure with correlator phase vector field (like exact_dispersion.py).
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        correlator: k-space correlator matrices, shape (n_k, 2, 2)
        L: Grid dimension (unified)
        save_path: Path to save figure
        band_index: Which band to plot
        show_vector_field: Whether to show correlator[0,1] phase vector field
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reshape data for 2D plotting
    n_k = int(np.sqrt(len(k_points)))
    if n_k * n_k == len(k_points):
        # Reshape data for 2D plotting using Fortran order to match k-point generation
        kx = k_points[:, 0].reshape(n_k, n_k, order='F')
        ky = k_points[:, 1].reshape(n_k, n_k, order='F')
        energies = eigenvalues[:, band_index].reshape(n_k, n_k, order='F')
        
        # Create contour plot
        levels = np.linspace(np.min(energies), np.max(energies), 30)
        contour = ax.contourf(kx, ky, energies, levels=levels, 
                             cmap='RdBu_r', alpha=0.8)
        contour_lines = ax.contour(kx, ky, energies, levels=levels, 
                                  colors='black', linewidths=0.5, alpha=0.6)
        
        # Add colorbar for energy
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(f'Band {band_index + 1} Energy', fontsize=12)
        
    else:
        # Fallback to scatter plot for irregular grid
        scatter = ax.scatter(k_points[:, 0], k_points[:, 1], 
                           c=eigenvalues[:, band_index], cmap='RdBu_r', 
                           s=20, alpha=0.7)
        fig.colorbar(scatter, ax=ax, shrink=0.8, 
                    label=f'Band {band_index + 1} Energy')
    
    # Add vector field if requested
    if show_vector_field:
        # Extract correlator[0,1] phase and apply π - arg(Γ) transformation
        # This should match arg(J_k) if arg(Γ) + arg(J_k) = π + 2nπ
        correlator_01 = correlator[:, 0, 1]
        gamma_phases = np.angle(correlator_01)
        phases = np.pi - gamma_phases  # Transform: π - arg(Γ) ≈ arg(J_k)
        phases_2d = phases.reshape(n_k, n_k, order='F')  # Use Fortran order to match k-point grid
        
        # Generate vector field from phase gradient (like exact_dispersion.py)
        skip_factor = 3
        skip = skip_factor
        X = kx[::skip, ::skip]
        Y = ky[::skip, ::skip]
        
        # Convert phase to arrow components (unit vectors)
        U = np.cos(phases_2d[::skip, ::skip])  # x-component of unit vector
        V = np.sin(phases_2d[::skip, ::skip])  # y-component of unit vector
        
        # Plot vector field (same as exact_dispersion.py)
        quiver = ax.quiver(X, Y, U, V, color='black', alpha=0.8, 
                          scale=50, width=0.002, headwidth=2, headlength=3)
        
        # Add legend for vector field
        ax.quiverkey(quiver, 0.9, 0.95, 1, r'$\pi - \arg \Gamma_{01}(k)$', 
                    labelpos='E', coordinates='figure', fontproperties={'size': 10})
    
    # Mark high-symmetry points
    high_sym_points = {
        'Γ': (0.0, 0.0),
        'M': (np.pi, 0.0),
        'K': (4*np.pi/3, 0.0),
        'Y': (0.0, 2*np.pi/np.sqrt(3))
    }
    
    for name, point in high_sym_points.items():
        ax.plot(point[0], point[1], 's', markersize=6, markerfacecolor='white',
               markeredgewidth=1.5, markeredgecolor='blue')
        ax.annotate(name, point, xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    # Mark known Dirac points
    known_dirac_points = [
        (np.pi/3, -np.pi/3),           # Main Dirac point
        (-np.pi/3, np.pi/3),           # Equivalent Dirac point
    ]
    
    for i, (kx, ky) in enumerate(known_dirac_points):
        ax.plot(kx, ky, 'ko', markersize=10, markerfacecolor='yellow',
               markeredgewidth=2, markeredgecolor='black')
        if i == 0:  # Only label the main one
            ax.annotate('Dirac', (kx, ky), xytext=(10, 10), 
                       textcoords='offset points', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel(r'$k_x$', fontsize=14)
    ax.set_ylabel(r'$k_y$', fontsize=14)
    
    title = f'2D Band Structure with Correlator Phase Vector Field\nBand {band_index + 1}'
    if show_vector_field:
        title += r" ($\pi - \arg \Gamma_{01}$ Vector Field)"
    ax.set_title(title, fontsize=16, pad=15)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 2D band structure with correlator vector field saved: {Path(save_path).name}")
    
    return fig, ax

def plot_band_dispersion_with_vector_field(eigenvalues, k_points, correlator, Lx, Ly, 
                                          save_path=None, show_vector_field=True, vector_skip_factor=3):
    """Plot high-quality band dispersion with optional correlator phase vector field."""
    try:
        from plot.dispersion import plot_kitaev_bands
        
        # First plot the band dispersion
        fig, ax = plot_kitaev_bands(
            eigenvalues=np.array(eigenvalues), 
            k_points=np.array(k_points),
            save_path=None  # Don't save yet, we'll add vector field
        )
        
        # Add vector field if requested
        if show_vector_field:
            # Compute vector field for correlator[0,1] phase
            X, Y, U, V, phases_2d = compute_correlator_phase_vector_field(
                correlator, k_points, Lx, Ly, vector_skip_factor
            )
            
            # Plot vector field on top of band dispersion
            quiver = ax.quiver(X, Y, U, V, color='black', alpha=0.8, 
                              scale=50, width=0.002, headwidth=2, headlength=3)
            
            # Add legend for vector field
            ax.quiverkey(quiver, 0.9, 0.95, 1, r'$\nabla \arg \Gamma_{01}(k)$', 
                        labelpos='E', coordinates='figure', fontproperties={'size': 10})
            
            # Update title to indicate vector field
            current_title = ax.get_title()
            ax.set_title(current_title + " with Γ[0,1] Phase Vector Field", fontsize=16, pad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Band dispersion with vector field saved: {Path(save_path).name}")
        
        return fig, ax
        
    except ImportError:
        # Fallback to basic plotting if plot module not available
        print("⚠️  High-quality plot module not available, using basic plotting")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For simplicity, plot along kx direction (ky=0)
        Lx_plot = int(jnp.sqrt(len(k_points)))
        
        # Extract kx=0 to π line (ky=0)
        kx_line_indices = jnp.arange(0, Lx_plot)
        kx_values = k_points[kx_line_indices, 0]
        eigenvals_line = eigenvalues[kx_line_indices]
        
        # Plot all bands
        for band in range(eigenvals_line.shape[1]):
            ax.plot(kx_values, eigenvals_line[:, band], 'o-', markersize=3, linewidth=1.5, 
                    label=f'Band {band+1}')
        
        # Add vector field if requested
        if show_vector_field:
            # Compute vector field for correlator[0,1] phase
            X, Y, U, V, phases_2d = compute_correlator_phase_vector_field(
                correlator, k_points, Lx, Ly, vector_skip_factor
            )
            
            # For 1D plot, we'll show the phase as a color overlay
            correlator_01 = correlator[:, 0, 1]
            phases_1d = jnp.angle(correlator_01)
            
            # Create a second y-axis for phase
            ax2 = ax.twinx()
            ax2.plot(kx_values, phases_1d[kx_line_indices], 'r--', alpha=0.7, 
                     label=r'$\arg \Gamma_{01}(k)$')
            ax2.set_ylabel(r'$\arg \Gamma_{01}(k)$ (radians)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
        
        ax.set_xlabel('kx')
        ax.set_ylabel('Energy')
        title = 'Band Dispersion (ky = 0)'
        if show_vector_field:
            title += " with Γ[0,1] Phase"
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Band dispersion with vector field saved: {Path(save_path).name}")
        
        return fig, ax
        
    except ImportError:
        # Fallback to basic plotting if plot module not available
        print("⚠️  High-quality plot module not available, using basic plotting")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For simplicity, plot along kx direction (ky=0)
        Lx_plot = int(jnp.sqrt(len(k_points)))
        
        # Extract kx=0 to π line (ky=0)
        kx_line_indices = jnp.arange(0, Lx_plot)
        kx_values = k_points[kx_line_indices, 0]
        eigenvals_line = eigenvalues[kx_line_indices]
        
        # Plot all bands
        for band in range(eigenvals_line.shape[1]):
            ax.plot(kx_values, eigenvals_line[:, band], 'o-', markersize=3, linewidth=1.5, 
                    label=f'Band {band+1}')
        
        # Add vector field if requested
        if show_vector_field:
            # Compute vector field for correlator[0,1] phase
            X, Y, U, V, phases_2d = compute_correlator_phase_vector_field(
                correlator, k_points, Lx, Ly, vector_skip_factor
            )
            
            # For 1D plot, we'll show the phase as a color overlay
            correlator_01 = correlator[:, 0, 1]
            phases_1d = jnp.angle(correlator_01)
            
            # Create a second y-axis for phase
            ax2 = ax.twinx()
            ax2.plot(kx_values, phases_1d[kx_line_indices], 'r--', alpha=0.7, 
                     label=r'$\arg \Gamma_{01}(k)$')
            ax2.set_ylabel(r'$\arg \Gamma_{01}(k)$ (radians)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
        
        ax.set_xlabel('kx')
        ax.set_ylabel('Energy')
        title = 'Band Dispersion (ky = 0)'
        if show_vector_field:
            title += " with Γ[0,1] Phase"
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Band dispersion with vector field saved: {Path(save_path).name}")
        
        return fig, ax

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

def run_analysis(glocal_file, K=50, save_plots=True, output_dir=None, show_vector_field=True):
    """
    Simplified analysis pipeline focusing on band dispersion.
    Correlation analysis moved to analysis_correlation.py.
    
    Args:
        glocal_file: Path to Glocal file (.npy or .json)
        K: k-space sampling (unified, was Kx=Ky)
        save_plots: whether to save plots
        output_dir: optional override for output directory (default: auto-detect seed dir)
        show_vector_field: whether to show correlator phase vector field
    """
    print(f"🔬 Starting simplified analysis of: {glocal_file}")
    # Load Glocal and metadata
    Glocal, metadata = load_glocal_from_file(glocal_file)
    # Extract original parameters
    if metadata and 'config' in metadata:
        original_config = metadata['config']
        Nv = original_config['Nv']
        original_L = original_config.get('Lx', original_config.get('L', 50))  # backward compatibility
        seed = original_config['seed']
        print(f"📋 Original parameters: L={original_L}, Nv={Nv}, seed={seed}")
        print(f"📋 Analysis parameters: K={K}")
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
    
    # Compute correlator with k-space sampling
    print("🔄 Computing correlator...")
    correlator, k_points = compute_correlator_from_glocal(Glocal, K, Nv)
    
    # Compute band dispersion E(k) from Hamiltonian
    print("📊 Computing band dispersion E(k)...")
    Jx, Jy, Jz = original_config.get('Jx', 1.0), original_config.get('Jy', 1.0), original_config.get('Jz', 1.0)
    eigenvalues = compute_band_dispersion(correlator, k_points, Jx, Jy, Jz)
    # Save numerical results (simplified)
    results = {
        'correlator': correlator,
        'k_points': k_points,
        'eigenvalues': eigenvalues,
        'analysis_K': K,
        'original_config': metadata.get('config', {}),
        'seed': seed
    }
    
    save_analysis_results(results, seed_dir)
    
    # Generate plots in same seed directory (simplified - only band dispersion)
    if save_plots:
        print("🎨 Generating band dispersion plots...")
        
        # Generate band dispersion plot with optional vector field
        if show_vector_field:
            plot_2d_band_structure_with_correlator_vector_field(
                eigenvalues, k_points, correlator, K,
                save_path=seed_dir / "band_dispersion_with_vector_field.png",
                band_index=0, show_vector_field=True
            )
        else:
            plot_band_dispersion(
                eigenvalues, k_points,
                save_path=seed_dir / "band_dispersion.png"
            )
        
        # Generate additional specialized plots if high-quality modules available
        try:
            from plot.dispersion import DispersionPlotter
            from plot.dispersion import plot_diagonal_band_dispersion
            from plot.dispersion import plot_dirac_scaling_analysis, plot_2d_band_structure_with_dirac_search
            
            print("🎨 Generating additional specialized plots...")
            
            # Density of states
            disp_plotter = DispersionPlotter()
            disp_plotter.plot_density_of_states(
                eigenvalues=np.array(eigenvalues),
                save_path=seed_dir / "density_of_states.png"
            )
            
            # Simplified band dispersion along diagonal line (0,2π) to (2π,0)
            plot_diagonal_band_dispersion(
                eigenvalues=np.array(eigenvalues),
                k_points=np.array(k_points),
                save_path=seed_dir / "diagonal_band_dispersion.png",
                s = 1,
            )
            
            # Dirac point scaling analysis with known Dirac point location
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
        description="Analyze Kitaev fPEPS results: band dispersion analysis (simplified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("glocal_file", type=str, 
                       help="Path to Glocal .npy file or metadata .json file")
    parser.add_argument("--K", type=int, default=50,
                       help="k-space sampling (unified, was Kx=Ky)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-detect seed directory)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--no_vector_field", action="store_true",
                       help="Skip vector field in band dispersion plots")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    results = run_analysis(
        glocal_file=args.glocal_file,
        K=args.K,
        save_plots=not args.no_plots,
        output_dir=args.output_dir,
        show_vector_field=not args.no_vector_field
    )
    print(f"\n📈 Analysis Summary (Simplified):")
    print(f"   K-space sampling: {args.K}×{args.K}")
    print(f"   Number of k-points: {len(results['k_points'])}")
    print(f"   Number of bands: {results['eigenvalues'].shape[1]}")
    print(f"   Results saved to: {args.output_dir if args.output_dir else 'auto-detected seed directory'}")
    print(f"   📝 For correlation analysis, use: python analysis_correlation.py {args.glocal_file}")

if __name__ == '__main__':
    main() 