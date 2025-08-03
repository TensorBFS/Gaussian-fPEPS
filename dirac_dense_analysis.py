#!/usr/bin/env python3
"""
Dense Dirac point analysis for Kitaev fPEPS results.

This script performs extremely dense sampling around Dirac points
to analyze the scaling behavior with high precision.

Author: Kitaev-fPEPS Project
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

def kitaev_kernel_for_phase(k, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Kitaev honeycomb spin-1/2 energy kernel for phase calculation.
    Returns the J(k) complex value whose phase we want to plot.
    
    Args:
        k: Momentum vector (kx, ky)
        Jx, Jy, Jz: Kitaev coupling strengths
        
    Returns:
        J(k) = Jz - Jx*exp(ikx) - Jy*exp(iky)
    """
    kx, ky = k[0], k[1]
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    return Jk

def compute_Jk_phase_for_points(k_points, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Compute the phase of J(k) for given k-points.
    
    Args:
        k_points: Array of k-points, shape (n_k, 2)
        Jx, Jy, Jz: Kitaev coupling parameters
    
    Returns:
        Jk_phases: Phase of J(k), shape (n_k,)
        Jk_magnitudes: Magnitude of J(k), shape (n_k,)
    """
    def compute_single_Jk(k):
        """Compute J(k) for a single k-point."""
        Jk = kitaev_kernel_for_phase(k, Jx, Jy, Jz)
        phase = jnp.angle(Jk)
        magnitude = jnp.abs(Jk)
        return jnp.array([phase, magnitude])
    
    # Vectorized computation over all k-points
    Jk_data = vmap(compute_single_Jk)(k_points)
    phases = Jk_data[:, 0]
    magnitudes = Jk_data[:, 1]
    
    return phases, magnitudes

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
    Compute band dispersion E(k) using expectation values (following kitaev.py).
    
    For each k-point: E(k) = mean(real(dot(correlator, H_k)))
    This follows the exact implementation in kitaev.py for energy calculation.
    
    Args:
        correlator: k-space correlator matrices (from fPEPS optimization)
        k_points: corresponding k-points, shape (Lx*Ly, 2)
        Jx, Jy, Jz: Kitaev coupling parameters
    
    Returns:
        eigenvalues: Band energies E(k), shape (Lx*Ly, 2)
    """
    from kitaev import kitaev_kernel
    
    def compute_energy_for_k(corr_k, k):
        # Get the 2x2 Hamiltonian matrix for this k-point
        H_k = kitaev_kernel(k, Jx, Jy, Jz)
        # FIXED: Use element-wise multiplication and sum instead of dot product and mean
        # ⟨H⟩ = Σᵢⱼ Γᵢⱼ(k) hᵢⱼ(k), not dot product
        energy = jnp.sum(jnp.real(corr_k * H_k))
        return jnp.array([energy, -energy])
    
    # Vectorized computation over all k-points
    eigenvals = vmap(compute_energy_for_k)(correlator, k_points)
    
    return eigenvals

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

# ============================================================================
# Dirac Point Analysis
# ============================================================================

def generate_dense_dirac_grid(dirac_point, radius=0.1, n_rings=10, n_angles=32):
    """
    Generate systematic k-point grid around Dirac point using concentric rings.
    
    Args:
        dirac_point: (kx, ky) coordinates of Dirac point
        radius: Maximum radius around Dirac point
        n_rings: Number of concentric rings
        n_angles: Number of angular points per ring
    
    Returns:
        k_points: Systematic grid of k-points around Dirac point
        ring_info: Dictionary with ring and angle information for each point
    """
    k_points = []
    ring_indices = []
    angle_indices = []
    
    # Add center point (Dirac point itself)
    k_points.append([dirac_point[0], dirac_point[1]])
    ring_indices.append(0)
    angle_indices.append(0)
    
    # Generate concentric rings
    for ring_idx in range(1, n_rings + 1):
        # Calculate radius for this ring (equal spacing)
        ring_radius = radius * ring_idx / n_rings
        
        # Generate points at equal angular intervals
        for angle_idx in range(n_angles):
            angle = 2 * np.pi * angle_idx / n_angles
            
            # Convert to Cartesian coordinates
            kx = dirac_point[0] + ring_radius * np.cos(angle)
            ky = dirac_point[1] + ring_radius * np.sin(angle)
            
            k_points.append([kx, ky])
            ring_indices.append(ring_idx)
            angle_indices.append(angle_idx)
    
    k_points = np.array(k_points)
    
    # Return additional information about the grid structure
    ring_info = {
        'ring_indices': np.array(ring_indices),
        'angle_indices': np.array(angle_indices),
        'n_rings': n_rings,
        'n_angles': n_angles,
        'total_points': len(k_points)
    }
    
    return k_points, ring_info

def analyze_dirac_scaling_dense(glocal_file, dirac_point=(np.pi/3, -np.pi/3), 
                               radius=0.1, n_rings=10, n_angles=32, save_path=None):
    """
    Perform systematic analysis around Dirac point using concentric rings.
    
    Args:
        glocal_file: Path to Glocal file
        dirac_point: Dirac point coordinates
        radius: Analysis radius
        n_rings: Number of concentric rings
        n_angles: Number of angular points per ring
        save_path: Path to save results
    """
    print(f"🔬 Starting systematic Dirac analysis...")
    print(f"📋 Dirac point: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})")
    print(f"📋 Radius: {radius}, Rings: {n_rings}, Angles per ring: {n_angles}")
    print(f"📋 Total points: {1 + n_rings * n_angles}")
    
    # Load Glocal and metadata
    Glocal, metadata = load_glocal_from_file(glocal_file)
    
    # Extract parameters
    if metadata and 'config' in metadata:
        original_config = metadata['config']
        Nv = original_config['Nv']
        Jx, Jy, Jz = original_config.get('Jx', 1.0), original_config.get('Jy', 1.0), original_config.get('Jz', 1.0)
        print(f"📋 Parameters: Nv={Nv}, Jx={Jx}, Jy={Jy}, Jz={Jz}")
    else:
        raise ValueError("Could not extract parameters from metadata")
    
    # Generate systematic k-point grid around Dirac point
    print("🔄 Generating systematic k-point grid around Dirac point...")
    dense_k_points, ring_info = generate_dense_dirac_grid(dirac_point, radius, n_rings, n_angles)
    n_points = ring_info['total_points']
    
    # Compute correlator for dense grid
    print("🔄 Computing correlator for dense grid...")
    correlator, _ = compute_correlator_from_glocal(Glocal, 100, 100, Nv)  # Use large grid for interpolation
    
    # Interpolate correlator to dense grid
    from scipy.interpolate import griddata
    
    # Generate reference grid for interpolation
    ref_k_points = batched_k(100, 100)
    
    # Interpolate each component of the correlator
    dense_correlator = np.zeros((n_points, 2*NF, 2*NF), dtype=complex)
    
    for i in range(2*NF):
        for j in range(2*NF):
            dense_correlator[:, i, j] = griddata(
                ref_k_points, correlator[:, i, j], dense_k_points,
                method='linear', fill_value=np.nan
            )
    
    # Compute band dispersion for dense grid
    print("📊 Computing band dispersion for dense grid...")
    eigenvalues = compute_band_dispersion(dense_correlator, dense_k_points, Jx, Jy, Jz)
    
    # Calculate distances from Dirac point
    distances = np.sqrt((dense_k_points[:, 0] - dirac_point[0])**2 + 
                       (dense_k_points[:, 1] - dirac_point[1])**2)
    
    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_energies = np.abs(eigenvalues[sorted_indices, 0])  # Use first band
    
    # Remove NaN values
    valid_mask = ~np.isnan(sorted_energies)
    sorted_distances = sorted_distances[valid_mask]
    sorted_energies = sorted_energies[valid_mask]
    
    print(f"📊 Valid data points: {len(sorted_energies)}")
    print(f"📊 Distance range: [{np.min(sorted_distances):.4f}, {np.max(sorted_distances):.4f}]")
    print(f"📊 Energy range: [{np.min(sorted_energies):.4f}, {np.max(sorted_energies):.4f}]")
    
    # Create analysis plots
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create 2D plot with three subplots: energy, J(k) phase, and Gamma[0,1] phase
        fig_2d, (ax_energy, ax_jk_phase, ax_gamma_phase) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Calculate dkx and dky relative to Dirac point
        dkx = dense_k_points[:, 0] - dirac_point[0]
        dky = dense_k_points[:, 1] - dirac_point[1]
        
        # Compute J(k) phases for the dense grid
        print("🔄 Computing J(k) phases for dense grid...")
        Jk_phases, Jk_magnitudes = compute_Jk_phase_for_points(dense_k_points, Jx, Jy, Jz)
        
        # Plot 1: Energy distribution
        energy_values = np.abs(eigenvalues[:, 0])  # Use first band
        
        # Create scatter plot with energy as color
        scatter_energy = ax_energy.scatter(dkx, dky, c=energy_values, cmap='viridis', 
                                          s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add colorbar for energy
        cbar_energy = plt.colorbar(scatter_energy, ax=ax_energy)
        cbar_energy.set_label(r'$|E(k)|$', fontsize=14)
        
        # Add Dirac point marker
        ax_energy.scatter(0, 0, c='red', s=100, marker='*', edgecolors='black', 
                         linewidth=1, label='Dirac Point', zorder=5)
        
        # Add circle showing analysis radius
        circle_energy = plt.Circle((0, 0), radius, fill=False, color='red', 
                                  linestyle='--', linewidth=2, label=f'Analysis radius = {radius}')
        ax_energy.add_patch(circle_energy)
        
        # Set labels and title for energy plot
        ax_energy.set_xlabel(r'$\Delta k_x$', fontsize=14)
        ax_energy.set_ylabel(r'$\Delta k_y$', fontsize=14)
        ax_energy.set_title(f'Energy Distribution |E(k)|\nDirac Point: ({dirac_point[0]:.2f}, {dirac_point[1]:.2f})', 
                           fontsize=14)
        
        # Set equal aspect ratio and limits for energy plot
        ax_energy.set_aspect('equal')
        ax_energy.set_xlim(-radius*1.1, radius*1.1)
        ax_energy.set_ylim(-radius*1.1, radius*1.1)
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend(fontsize=12)
        
        # Plot 2: J(k) Phase distribution using arrows
        # Convert J(k) phase to arrow components (unit vectors)
        U_jk = np.cos(Jk_phases)  # x-component of unit vector
        V_jk = np.sin(Jk_phases)  # y-component of unit vector
        
        # Use fixed arrow length for clearer phase visualization
        fixed_arrow_length = 0.1 * radius
        U_jk_scaled = U_jk * fixed_arrow_length
        V_jk_scaled = V_jk * fixed_arrow_length
        
        # First plot points to show sampling locations (colored by J(k) magnitude)
        ax_jk_phase.scatter(dkx, dky, c=Jk_magnitudes, cmap='plasma', 
                           s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Create quiver plot with arrows showing J(k) phase direction
        quiver_jk = ax_jk_phase.quiver(dkx, dky, U_jk_scaled, V_jk_scaled, 
                                      Jk_magnitudes, cmap='plasma', 
                                      scale=1, scale_units='xy', angles='xy',
                                      alpha=0.9, width=0.002, headwidth=3, headlength=4)
        
        # Add colorbar for J(k) magnitude
        cbar_jk = plt.colorbar(quiver_jk, ax=ax_jk_phase)
        cbar_jk.set_label(r'$|J(k)|$ magnitude', fontsize=14)
        
        # Add Dirac point marker
        ax_jk_phase.scatter(0, 0, c='red', s=100, marker='*', edgecolors='black', 
                           linewidth=1, label='Dirac Point', zorder=5)
        
        # Add circle showing analysis radius
        circle_jk = plt.Circle((0, 0), radius, fill=False, color='red', 
                              linestyle='--', linewidth=2, label=f'Analysis radius = {radius}')
        ax_jk_phase.add_patch(circle_jk)
        
        # Set labels and title for J(k) phase plot
        ax_jk_phase.set_xlabel(r'$\Delta k_x$', fontsize=14)
        ax_jk_phase.set_ylabel(r'$\Delta k_y$', fontsize=14)
        ax_jk_phase.set_title(r'$J(k)$ Phase Vectors (arrows show $\arg[J(k)]$)' + '\nfrom Kitaev Hamiltonian kernel', 
                             fontsize=14)
        
        # Set equal aspect ratio and limits for J(k) phase plot
        ax_jk_phase.set_aspect('equal')
        ax_jk_phase.set_xlim(-radius*1.1, radius*1.1)
        ax_jk_phase.set_ylim(-radius*1.1, radius*1.1)
        ax_jk_phase.grid(True, alpha=0.3)
        ax_jk_phase.legend(fontsize=12)
        
        # Plot 3: Correlator Gamma[0,1] Phase distribution using arrows
        # Extract Gamma[0,1] from dense correlator
        gamma_01 = dense_correlator[:, 0, 1]  # Complex values
        gamma_01_phases = np.angle(gamma_01)  # Phase of Gamma[0,1]
        gamma_01_magnitudes = np.abs(gamma_01)  # Magnitude of Gamma[0,1]
        
        # Convert phase to arrow components (unit vectors)
        U_gamma = np.cos(gamma_01_phases)  # x-component of unit vector
        V_gamma = np.sin(gamma_01_phases)  # y-component of unit vector
        
        # Use fixed arrow length for clearer phase visualization
        U_gamma_scaled = U_gamma * fixed_arrow_length
        V_gamma_scaled = V_gamma * fixed_arrow_length
        
        # First plot points to show sampling locations
        ax_gamma_phase.scatter(dkx, dky, c=gamma_01_magnitudes, cmap='plasma', 
                              s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Create quiver plot with arrows showing phase direction
        quiver_gamma = ax_gamma_phase.quiver(dkx, dky, U_gamma_scaled, V_gamma_scaled, 
                                           gamma_01_magnitudes, cmap='plasma', 
                                           scale=1, scale_units='xy', angles='xy',
                                           alpha=0.9, width=0.002, headwidth=3, headlength=4)
        
        # Add colorbar for magnitude
        cbar_gamma = plt.colorbar(quiver_gamma, ax=ax_gamma_phase)
        cbar_gamma.set_label(r'$|\Gamma_{01}(k)|$ magnitude', fontsize=14)
        
        # Add Dirac point marker
        ax_gamma_phase.scatter(0, 0, c='red', s=100, marker='*', edgecolors='black', 
                              linewidth=1, label='Dirac Point', zorder=5)
        
        # Add circle showing analysis radius
        circle_gamma = plt.Circle((0, 0), radius, fill=False, color='red', 
                                 linestyle='--', linewidth=2, label=f'Analysis radius = {radius}')
        ax_gamma_phase.add_patch(circle_gamma)
        
        # Set labels and title for Gamma phase plot
        ax_gamma_phase.set_xlabel(r'$\Delta k_x$', fontsize=14)
        ax_gamma_phase.set_ylabel(r'$\Delta k_y$', fontsize=14)
        ax_gamma_phase.set_title(r'$\Gamma_{01}(k)$ Phase Vectors (arrows show $\arg[\Gamma_{01}(k)]$)' + '\nfrom fPEPS correlator matrix', 
                                fontsize=14)
        
        # Set equal aspect ratio and limits for Gamma phase plot
        ax_gamma_phase.set_aspect('equal')
        ax_gamma_phase.set_xlim(-radius*1.1, radius*1.1)
        ax_gamma_phase.set_ylim(-radius*1.1, radius*1.1)
        ax_gamma_phase.grid(True, alpha=0.3)
        ax_gamma_phase.legend(fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save 2D plot
        save_path_2d = save_path.with_name(save_path.stem + '_2d' + save_path.suffix)
        plt.savefig(save_path_2d, dpi=300, bbox_inches='tight')
        print(f"🎨 2D systematic Dirac analysis (energy + J(k) phase + Gamma[0,1] phase) saved: {save_path_2d.name}")
        plt.close()
        
        # Plot 1: dK^2 vs E(k)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sample_dk2 = sorted_distances**2
        sample_energies = sorted_energies
        
        # Plot 1: dK^2 vs E(k)
        ax1.scatter(sample_dk2, sample_energies, c='blue', s=20, alpha=0.7, 
                    label=f'Data points (n={len(sample_energies)})')
        
        # Fit linear relationship: E = α * dK^2
        if len(sample_dk2) > 5:
            try:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(sample_dk2, sample_energies)
                
                # Plot fitted line
                x_fit = np.linspace(0, np.max(sample_dk2), 100)
                y_fit = slope * x_fit + intercept
                ax1.plot(x_fit, y_fit, 'r--', linewidth=2, 
                        label=f'Linear fit: E = {slope:.3f} × dK² + {intercept:.3f}\nR² = {r_value**2:.3f}')
                
                # Add fit statistics
                ax1.text(0.05, 0.95, f'Slope = {slope:.3f} ± {std_err:.3f}\nR² = {r_value**2:.3f}', 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                        verticalalignment='top')
                
                print(f"📊 Linear fit: slope = {slope:.3f} ± {std_err:.3f}, R² = {r_value**2:.3f}")
                
            except ImportError:
                print("⚠️  scipy not available for fitting")
        
        ax1.set_xlabel(r'$(\Delta k)^2$', fontsize=14)
        ax1.set_ylabel(r'$|E(k)|$', fontsize=14)
        ax1.set_title(f'Systematic Dirac Scaling: dK² vs E(k)\nDirac point: ({dirac_point[0]:.2f}, {dirac_point[1]:.2f})', 
                      fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Plot 2: Power law analysis (log-log plot)
        # Use dK instead of dK^2 for power law analysis
        sample_dk = np.sqrt(sample_dk2)
        
        # Remove zero values for log plot
        nonzero_mask = (sample_dk > 0) & (sample_energies > 0)
        if np.sum(nonzero_mask) > 5:
            log_dk = np.log(sample_dk[nonzero_mask])
            log_energy = np.log(sample_energies[nonzero_mask])
            
            ax2.scatter(log_dk, log_energy, c='green', s=20, alpha=0.7, 
                       label=f'Data points (n={len(log_energy)})')
            
            # Fit power law: E = A * (dK)^n
            try:
                from scipy.stats import linregress
                slope_power, intercept_power, r_value_power, p_value_power, std_err_power = linregress(log_dk, log_energy)
                
                # Plot fitted line
                x_fit_log = np.linspace(np.min(log_dk), np.max(log_dk), 100)
                y_fit_log = slope_power * x_fit_log + intercept_power
                ax2.plot(x_fit_log, y_fit_log, 'r--', linewidth=2,
                        label=f'Power law fit: E ∝ (dK)^{slope_power:.2f}\nR² = {r_value_power**2:.3f}')
                
                # Add interpretation
                if abs(slope_power - 1.0) < 0.2:
                    interpretation = "Linear dispersion (Dirac cone)"
                elif abs(slope_power - 2.0) < 0.2:
                    interpretation = "Quadratic dispersion"
                else:
                    interpretation = f"Non-standard scaling (n = {slope_power:.2f})"
                
                ax2.text(0.05, 0.95, f'Power law exponent = {slope_power:.2f} ± {std_err_power:.2f}\n{interpretation}', 
                        transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                        verticalalignment='top')
                
                print(f"📊 Power law fit: exponent = {slope_power:.2f} ± {std_err_power:.2f}, R² = {r_value_power**2:.3f}")
                print(f"📊 Interpretation: {interpretation}")
                
            except ImportError:
                print("⚠️  scipy not available for power law fitting")
        
        ax2.set_xlabel(r'$\ln(\Delta k)$', fontsize=14)
        ax2.set_ylabel(r'$\ln(|E(k)|)$', fontsize=14)
        ax2.set_title('Systematic Power Law Analysis: ln(dK) vs ln(E)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Systematic Dirac analysis saved: {save_path.name}")
        
        plt.close()
    
    # Extract Gamma[0,1] phase information for saved results
    gamma_01_final = dense_correlator[:, 0, 1]
    gamma_01_phases_final = np.angle(gamma_01_final)
    gamma_01_magnitudes_final = np.abs(gamma_01_final)
    
    # Save numerical results
    results = {
        'dirac_point': dirac_point,
        'radius': radius,
        'n_rings': n_rings,
        'n_angles': n_angles,
        'n_points': n_points,
        'ring_info': ring_info,
        'distances': sorted_distances,
        'energies': sorted_energies,
        'k_points': dense_k_points,
        'eigenvalues': eigenvalues,
        'dkx': dense_k_points[:, 0] - dirac_point[0],
        'dky': dense_k_points[:, 1] - dirac_point[1],
        'energy_2d': np.abs(eigenvalues[:, 0]),
        'jk_phases': np.array(Jk_phases),
        'jk_magnitudes': np.array(Jk_magnitudes),
        'gamma_01_phases': np.array(gamma_01_phases_final),
        'gamma_01_magnitudes': np.array(gamma_01_magnitudes_final),
        'gamma_01_complex': np.array(gamma_01_final),
        'parameters': {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}
    }
    
    # Save as numpy file
    np.save(save_path.with_suffix('.npy'), results)
    print(f"💾 Numerical results saved: {save_path.with_suffix('.npy').name}")
    
    return results

# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic Dirac point analysis for Kitaev fPEPS results using concentric rings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("glocal_file", type=str, 
                       help="Path to Glocal .npy file or metadata .json file")
    parser.add_argument("--dirac_x", type=float, default=np.pi/3,
                       help="Dirac point x coordinate")
    parser.add_argument("--dirac_y", type=float, default=-np.pi/3,
                       help="Dirac point y coordinate")
    parser.add_argument("--radius", type=float, default=0.001,
                       help="Analysis radius around Dirac point")
    parser.add_argument("--n_rings", type=int, default=10,
                       help="Number of concentric rings")
    parser.add_argument("--n_angles", type=int, default=32,
                       help="Number of angular points per ring")
    parser.add_argument("--output", type=str, default="dirac_dense_analysis.png",
                       help="Output file path")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    dirac_point = (args.dirac_x, args.dirac_y)
    
    results = analyze_dirac_scaling_dense(
        glocal_file=args.glocal_file,
        dirac_point=dirac_point,
        radius=args.radius,
        n_rings=args.n_rings,
        n_angles=args.n_angles,
        save_path=args.output
    )
    
    print(f"\n📈 Systematic Dirac Analysis Summary:")
    print(f"   Dirac point: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})")
    print(f"   Analysis radius: {args.radius}")
    print(f"   Concentric rings: {args.n_rings}")
    print(f"   Angular points per ring: {args.n_angles}")
    print(f"   Total sampling points: {results['n_points']}")
    print(f"   Valid data points: {len(results['energies'])}")
    print(f"   Results saved to: {args.output}")

if __name__ == '__main__':
    main() 