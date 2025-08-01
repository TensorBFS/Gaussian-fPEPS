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

def compute_Jk_phase(k_points, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Compute the phase of J(k) for vector field plotting.
    
    Args:
        k_points: Array of k-points, shape (n_k, 2)
        Jx, Jy, Jz: Kitaev coupling parameters
    
    Returns:
        Jk_phases: Phase of J(k), shape (n_k,)
        Jk_magnitudes: Magnitude of J(k), shape (n_k,)
    """
    def compute_single_Jk(k):
        """Compute J(k) for a single k-point."""
        kx, ky = k[0], k[1]
        Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
        phase = jnp.angle(Jk)
        magnitude = jnp.abs(Jk)
        return jnp.array([phase, magnitude])
    
    # Vectorized computation over all k-points
    Jk_data = vmap(compute_single_Jk)(k_points)
    phases = Jk_data[:, 0]
    magnitudes = Jk_data[:, 1]
    
    return phases, magnitudes

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
# Vector Field Generation
# ============================================================================

def generate_vector_field(kx_mesh, ky_mesh, phases, magnitudes, skip_factor=1):
    """
    Generate vector field data for plotting.
    
    Args:
        kx_mesh, ky_mesh: Coordinate meshgrids
        phases: Phase values at each point
        magnitudes: Magnitude values at each point
        skip_factor: Factor to skip points for cleaner vector field
    
    Returns:
        X, Y: Vector field coordinates
        U, V: Vector field components
    """
    # Reshape to 2D
    phases_2d = phases.reshape(kx_mesh.shape)
    magnitudes_2d = magnitudes.reshape(kx_mesh.shape)
    
    # Skip points for cleaner vector field
    skip = skip_factor
    X = kx_mesh[::skip, ::skip]
    Y = ky_mesh[::skip, ::skip]
    
    # Compute gradient of phase for vector field
    grad_phase_x = np.gradient(phases_2d, axis=1)[::skip, ::skip]
    grad_phase_y = np.gradient(phases_2d, axis=0)[::skip, ::skip]
    
    # Normalize vectors
    norm = np.sqrt(grad_phase_x**2 + grad_phase_y**2)
    U = grad_phase_x / (norm + 1e-10)
    V = grad_phase_y / (norm + 1e-10)
    
    return X, Y, U, V

# ============================================================================
# 2D Plotting Function with Vector Field
# ============================================================================

def plot_kitaev_exact_2d_with_vector_field(Jx=1.0, Jy=1.0, Jz=1.0, band_index=0,
                                           kx_range=(-np.pi, np.pi), ky_range=(-np.pi, np.pi),
                                           n_points=100, save_path=None, title=None,
                                           show_vector_field=True, vector_skip_factor=3):
    """
    Plot 2D contour map of exact Kitaev band energies with optional vector field.
    
    Args:
        Jx, Jy, Jz: Kitaev coupling parameters
        band_index: Which band to plot (0 or 1)
        kx_range, ky_range: k-space ranges
        n_points: Grid resolution
        save_path: Path to save the figure
        title: Custom title
        show_vector_field: Whether to show J(k) phase vector field
        vector_skip_factor: Factor to skip points for vector field
    
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
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create contour plot
    levels = np.linspace(np.min(band_energies), np.max(band_energies), 20)
    contour = ax.contourf(kx_mesh, ky_mesh, band_energies, levels=levels, 
                         cmap='RdBu_r', alpha=0.8)
    contour_lines = ax.contour(kx_mesh, ky_mesh, band_energies, levels=levels, 
                              colors='black', linewidths=0.5, alpha=0.6)
    
    # Add vector field if requested
    if show_vector_field:
        # Compute J(k) phase and magnitude
        Jk_phases, Jk_magnitudes = compute_Jk_phase(k_grid, Jx, Jy, Jz)
        
        # Generate vector field
        X, Y, U, V = generate_vector_field(kx_mesh, ky_mesh, Jk_phases, Jk_magnitudes, 
                                          vector_skip_factor)
        
        # Plot vector field
        quiver = ax.quiver(X, Y, U, V, color='black', alpha=0.8, 
                          scale=50, width=0.002, headwidth=2, headlength=3)
        
        # Add legend for vector field
        ax.quiverkey(quiver, 0.9, 0.95, 1, r'$\nabla \arg J(k)$', 
                    labelpos='E', coordinates='figure', fontproperties={'size': 10})
    
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
        vector_field_text = " with J(k) Phase Vector Field" if show_vector_field else ""
        ax.set_title(f'Kitaev Model: Band {band_index + 1} Energy Surface{vector_field_text} (Jx={Jx}, Jy={Jy}, Jz={Jz})', 
                    fontsize=16, pad=15)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"🎨 2D band structure with vector field saved: {Path(save_path).name}")
    
    return fig, ax, band_energies, kx_mesh, ky_mesh

# Keep the original function for backward compatibility
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
    return plot_kitaev_exact_2d_with_vector_field(
        Jx=Jx, Jy=Jy, Jz=Jz, band_index=band_index,
        kx_range=kx_range, ky_range=ky_range, n_points=n_points,
        save_path=save_path, title=title, show_vector_field=False
    )

# ============================================================================
# Main Function for Testing
# ============================================================================

if __name__ == "__main__":
    print("🎨 Exact Kitaev 2D Dispersion Plotter with Vector Field")
    print("📊 Computing exact band structure from Hamiltonian kernel...")
    
    # Example: Plot 2D energy surface for both bands with vector field
    for band_idx in [0, 1]:
        fig, ax, band_energies, kx_mesh, ky_mesh = plot_kitaev_exact_2d_with_vector_field(
            Jx=1.0, Jy=1.0, Jz=1.0,
            band_index=band_idx,
            show_vector_field=True,
            save_path=f"kitaev_exact_2d_band{band_idx+1}_with_vector_field.png"
        )
        print(f"✅ Generated Band {band_idx+1} energy surface with J(k) phase vector field")
    
    print("🎉 All 2D plots with vector fields generated!")
