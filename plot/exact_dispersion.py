#!/usr/bin/env python3
"""
Exact 2D dispersion plotting for Kitaev honeycomb spin-1/2 model.

This module computes and plots the EXACT band dispersion E(k) 
directly from the Kitaev Hamiltonian kernel without any approximation.

Physics Foundation:
- Based on Kitaev's exact solution for the honeycomb model
- Uses Majorana representation with antisymmetric Hamiltonian matrix
- For 2x2 antisymmetric matrix: eigenvalues are ±|H[0,1]|

Author: Kitaev-fPEPS Project
Style: Physical Review Letters computational physics standards
"""

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Enable double precision for exact calculations
jax.config.update("jax_enable_x64", True)

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 12,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# ============================================================================
# Core Physics: Exact Kitaev Dispersion
# ============================================================================

def kitaev_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Kitaev honeycomb spin-1/2 energy kernel in Majorana basis.
    
    This is the EXACT Hamiltonian matrix for the Kitaev model:
    h(k) = [ 0     J(k) ]
           [-J(k)   0   ]
    
    where J(k) = Jz - Jx*exp(ikx) - Jy*exp(iky)
    
    Args:
        k: Momentum vector (kx, ky)
        Jx, Jy, Jz: Kitaev coupling strengths
        
    Returns:
        2x2 antisymmetric matrix h(k) in Majorana basis
    """
    kx, ky = k[0], k[1]
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    return jnp.array([[0, Jk], [-Jk, 0]]) / 4.0

def compute_exact_dispersion(k_points, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Compute EXACT band dispersion E(k) from Kitaev Hamiltonian.
    
    For 2x2 antisymmetric matrix, eigenvalues are ±|H[0,1]|
    
    Args:
        k_points: Array of k-points, shape (n_k, 2)
        Jx, Jy, Jz: Kitaev coupling parameters
    
    Returns:
        eigenvalues: True band energies E(k), shape (n_k, 2)
    """
    def compute_single_point(k):
        """Compute bands for a single k-point."""
        H_k = kitaev_kernel(k, Jx, Jy, Jz)
        # For 2x2 antisymmetric matrix: eigenvalues are ±|H[0,1]|
        Ek = jnp.abs(H_k[0,1])
        return jnp.array([Ek, -Ek])
    
    # Vectorized computation over all k-points
    Eks = vmap(compute_single_point)(k_points)
    
    return Eks

# ============================================================================
# 2D Grid Generation
# ============================================================================

def generate_2d_grid(kx_range=(-np.pi, np.pi), ky_range=(-np.pi, np.pi), n_points=100):
    """
    Generate 2D k-point grid for contour plots.
    
    Args:
        kx_range: (kx_min, kx_max) range for kx
        ky_range: (ky_min, ky_max) range for ky
        n_points: Number of points along each direction
    
    Returns:
        k_grid: 2D grid of k-points, shape (n_points^2, 2)
        kx_mesh, ky_mesh: Meshgrids for plotting
    """
    kx = np.linspace(kx_range[0], kx_range[1], n_points)
    ky = np.linspace(ky_range[0], ky_range[1], n_points)
    
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    k_grid = np.column_stack([kx_mesh.ravel(), ky_mesh.ravel()])
    
    return k_grid, kx_mesh, ky_mesh

# ============================================================================
# 2D Plotting Function
# ============================================================================

def plot_kitaev_exact_2d(Jx=1.0, Jy=1.0, Jz=1.0, band_index=0,
                         kx_range=(-np.pi, np.pi), ky_range=(-np.pi, np.pi),
                         n_points=100, save_path=None, title=None):
    """
    Plot 2D contour map of exact Kitaev band energies.
    
    Args:
        Jx, Jy, Jz: Kitaev coupling parameters
        band_index: Which band to plot (0 or 1)
        kx_range, ky_range: k-space ranges
        n_points: Grid resolution
        save_path: Path to save the figure
        title: Custom title
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
        band_energies: 2D energy map
        kx_mesh, ky_mesh: Coordinate meshgrids
    """
    # Generate 2D k-grid
    k_grid, kx_mesh, ky_mesh = generate_2d_grid(kx_range, ky_range, n_points)
    
    # Compute exact dispersion
    eigenvalues = compute_exact_dispersion(k_grid, Jx, Jy, Jz)
    
    # Reshape for contour plotting
    band_energies = eigenvalues[:, band_index].reshape(n_points, n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    levels = np.linspace(np.min(band_energies), np.max(band_energies), 20)
    contour = ax.contourf(kx_mesh, ky_mesh, band_energies, levels=levels, 
                         cmap='RdBu_r', alpha=0.8)
    contour_lines = ax.contour(kx_mesh, ky_mesh, band_energies, levels=levels, 
                              colors='black', linewidths=0.5, alpha=0.6)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(f'Band {band_index + 1} Energy (eV)', fontsize=12)
    
    # Mark high-symmetry points
    high_sym_points = {
        'Γ': np.array([0.0, 0.0]),
        'M': np.array([np.pi, 0.0]),
        'K': np.array([4*np.pi/3, 0.0]),
        'Y': np.array([0.0, 2*np.pi/np.sqrt(3)])
    }
    
    for name, point in high_sym_points.items():
        if (kx_range[0] <= point[0] <= kx_range[1] and 
            ky_range[0] <= point[1] <= ky_range[1]):
            ax.plot(point[0], point[1], 'ko', markersize=8, markerfacecolor='white',
                   markeredgewidth=2, markeredgecolor='black')
            ax.annotate(name, (point[0], point[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlabel(r'$k_x$', fontsize=14)
    ax.set_ylabel(r'$k_y$', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=16, pad=15)
    else:
        ax.set_title(f'Kitaev Model: Band {band_index + 1} Energy Surface (Jx={Jx}, Jy={Jy}, Jz={Jz})', 
                    fontsize=16, pad=15)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"🎨 2D band structure saved: {Path(save_path).name}")
    
    return fig, ax, band_energies, kx_mesh, ky_mesh

# ============================================================================
# Main Function for Testing
# ============================================================================

if __name__ == "__main__":
    print("🎨 Exact Kitaev 2D Dispersion Plotter")
    print("📊 Computing exact band structure from Hamiltonian kernel...")
    
    # Example: Plot 2D energy surface for both bands
    for band_idx in [0, 1]:
        fig, ax, band_energies, kx_mesh, ky_mesh = plot_kitaev_exact_2d(
            Jx=1.0, Jy=1.0, Jz=1.0,
            band_index=band_idx,
            save_path=f"kitaev_exact_2d_band{band_idx+1}.png"
        )
        print(f"✅ Generated Band {band_idx+1} energy surface")
    
    print("🎉 All 2D plots generated!")
