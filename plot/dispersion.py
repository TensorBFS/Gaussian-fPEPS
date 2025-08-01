#!/usr/bin/env python3
"""
High-quality scientific plotting for band dispersion relations.
PRL-level publication-ready figures for computational physics.

Author: Kitaev-fPEPS Project
Style: Physical Review Letters computational physics standards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 12,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.major.size': 6,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.major.width': 1.2,
    'ytick.minor.width': 0.8,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 0.9,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Define high-symmetry points for honeycomb lattice (matching exact_dispersion.py)
HONEYCOMB_HIGH_SYMMETRY = {
    'Γ': np.array([0.0, 0.0]),
    'M': np.array([np.pi, 0.0]),
    'K': np.array([4*np.pi/3, 0.0]),
    'Y': np.array([0.0, 2*np.pi/np.sqrt(3)])
}

# Default high-symmetry path for honeycomb lattice
DEFAULT_PATH = ['Γ', 'M', 'K', 'Γ']

def generate_high_symmetry_path(points_dict=HONEYCOMB_HIGH_SYMMETRY, 
                               path=DEFAULT_PATH, n_points=100):
    """
    Generate k-points along high-symmetry path.
    
    Args:
        points_dict: Dictionary of high-symmetry points
        path: List of point names defining the path
        n_points: Number of points per segment
    
    Returns:
        k_path: Array of k-points along path
        k_distances: Cumulative distances along path
        tick_positions: Positions of high-symmetry points
        tick_labels: Labels for high-symmetry points
    """
    k_path = []
    k_distances = []
    tick_positions = [0.0]
    tick_labels = [path[0]]
    
    current_distance = 0.0
    
    for i in range(len(path) - 1):
        start_point = points_dict[path[i]]
        end_point = points_dict[path[i + 1]]
        
        # Generate points along this segment
        t = np.linspace(0, 1, n_points, endpoint=False)
        segment_k = start_point[np.newaxis, :] + t[:, np.newaxis] * (end_point - start_point)[np.newaxis, :]
        
        # Calculate distances
        if i == 0:
            segment_distances = current_distance + t * np.linalg.norm(end_point - start_point)
        else:
            segment_distances = current_distance + t * np.linalg.norm(end_point - start_point)
        
        k_path.append(segment_k)
        k_distances.append(segment_distances)
        
        # Update for next segment
        current_distance += np.linalg.norm(end_point - start_point)
        tick_positions.append(current_distance)
        tick_labels.append(path[i + 1])
    
    k_path = np.concatenate(k_path, axis=0)
    k_distances = np.concatenate(k_distances)
    
    return k_path, k_distances, tick_positions, tick_labels

class DispersionPlotter:
    """High-quality plotter for band dispersion relations."""
    
    def __init__(self, figsize=(10, 8)):
        """
        Initialize the dispersion plotter.
        
        Args:
            figsize: Figure size in inches
        """
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))  # High-quality color scheme
    
    def interpolate_bands_to_path(self, eigenvalues, k_points, high_sym_path):
        """
        Interpolate band eigenvalues to high-symmetry path.
        
        Args:
            eigenvalues: Band energies, shape (n_k, n_bands)
            k_points: k-point grid, shape (n_k, 2)
            high_sym_path: k-points along high-symmetry path
        
        Returns:
            interpolated_bands: Band energies along path
        """
        from scipy.interpolate import griddata
        
        n_bands = eigenvalues.shape[1]
        n_path_points = len(high_sym_path)
        interpolated_bands = np.zeros((n_path_points, n_bands))
        
        for band in range(n_bands):
            interpolated_bands[:, band] = griddata(
                k_points, eigenvalues[:, band], high_sym_path, 
                method='linear', fill_value=np.nan
            )
        
        return interpolated_bands
    
    def plot_band_structure(self, eigenvalues, k_points, 
                           title=None, save_path=None,
                           fermi_level=0.0, energy_window=None,
                           high_sym_points=HONEYCOMB_HIGH_SYMMETRY,
                           high_sym_path=DEFAULT_PATH,
                           show_fermi=True, show_gap=True):
        """
        Plot high-quality band structure along high-symmetry path.
        
        Args:
            eigenvalues: Band energies, shape (n_k, n_bands)
            k_points: k-point grid, shape (n_k, 2)
            title: Custom title for the plot
            save_path: Path to save the figure
            fermi_level: Fermi energy level
            energy_window: [E_min, E_max] energy range to plot
            high_sym_points: Dictionary of high-symmetry points
            high_sym_path: Path through high-symmetry points
            show_fermi: Whether to show Fermi level
            show_gap: Whether to analyze and show band gap
        """
        # Generate high-symmetry path
        path_k, path_distances, tick_positions, tick_labels = generate_high_symmetry_path(
            high_sym_points, high_sym_path, n_points=200
        )
        
        # Interpolate bands to high-symmetry path
        path_bands = self.interpolate_bands_to_path(eigenvalues, k_points, path_k)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_bands = path_bands.shape[1]
        
        # Plot each band with high-quality styling
        for band in range(n_bands):
            band_energies = path_bands[:, band] - fermi_level
            
            # Remove NaN values for clean plotting
            valid_mask = ~np.isnan(band_energies)
            if np.any(valid_mask):
                ax.scatter(path_distances[valid_mask], band_energies[valid_mask],
                color=self.colors[band % len(self.colors)],
                s=5,  # 更小的点
                alpha=0.3,  # 高密区域自然形成“热图”效果
                marker='o',
                linewidths=0,
                label=f'Band {band + 1}' if n_bands <= 6 else None)
        
        # Add Fermi level
        if show_fermi:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, 
                      alpha=0.8, label='Fermi Level')
        
        # Analyze and show band gap
        if show_gap:
            self._analyze_band_gap(ax, path_bands, fermi_level)
        
        # Add vertical lines at high-symmetry points
        for pos in tick_positions:
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('', fontsize=14)  # No label, use tick labels instead
        ax.set_ylabel('Energy (eV)', fontsize=14)
        
        if title:
            ax.set_title(title, fontsize=16, pad=15)
        else:
            ax.set_title('Band Structure', fontsize=16, pad=15)
        
        # Set high-symmetry point labels
        ax.set_xticks(tick_positions)
        
        # Format labels with proper symbols
        formatted_labels = []
        for label in tick_labels:
            if label == 'Γ':
                formatted_labels.append('Γ')
            elif label == 'K\'':
                formatted_labels.append('K\'')
            else:
                formatted_labels.append(label)
        
        ax.set_xticklabels(formatted_labels, fontsize=14)
        
        # Set energy window if specified
        if energy_window:
            ax.set_ylim(energy_window[0] - fermi_level, energy_window[1] - fermi_level)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if n_bands <= 6:  # Only show legend for few bands
            ax.legend(fontsize=10, loc='best')
        
        # Minor ticks for professional appearance
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"🎨 High-quality band structure saved: {Path(save_path).name}")
        
        return fig, ax
    
    def _analyze_band_gap(self, ax, path_bands, fermi_level):
        """Analyze and highlight band gap if present."""
        # Find bands crossing Fermi level
        bands_relative = path_bands - fermi_level
        
        # Find valence and conduction bands
        valence_max = -np.inf
        conduction_min = np.inf
        
        for band in range(bands_relative.shape[1]):
            band_energies = bands_relative[:, band]
            valid_energies = band_energies[~np.isnan(band_energies)]
            
            if len(valid_energies) > 0:
                band_max = np.max(valid_energies)
                band_min = np.min(valid_energies)
                
                # Check if band crosses Fermi level
                if band_min <= 0 <= band_max:
                    continue  # Metallic band
                elif band_max < 0:
                    valence_max = max(valence_max, band_max)
                elif band_min > 0:
                    conduction_min = min(conduction_min, band_min)
        
        # Highlight band gap if present
        if valence_max != -np.inf and conduction_min != np.inf and conduction_min > valence_max:
            gap_size = conduction_min - valence_max
            
            # Shade the gap region
            ax.axhspan(valence_max, conduction_min, alpha=0.2, color='gray', 
                      label=f'Band Gap: {gap_size:.3f} eV')
            
            # Add gap annotation
            gap_center = (valence_max + conduction_min) / 2
            ax.annotate(f'Gap = {gap_size:.3f} eV', 
                       xy=(ax.get_xlim()[1] * 0.02, gap_center),
                       fontsize=11, bbox=dict(boxstyle="round,pad=0.3", 
                                            facecolor='white', alpha=0.8))
    
    def plot_density_of_states(self, eigenvalues, save_path=None, 
                              fermi_level=0.0, energy_window=None, 
                              broadening=0.1, n_points=1000):
        """
        Plot density of states from band eigenvalues.
        
        Args:
            eigenvalues: Band energies, shape (n_k, n_bands)
            save_path: Path to save figure
            fermi_level: Fermi energy level
            energy_window: [E_min, E_max] energy range
            broadening: Gaussian broadening parameter
            n_points: Number of energy points for DOS
        """
        fig, ax = plt.subplots(figsize=(6, 8))
        
        # Flatten all eigenvalues
        all_energies = eigenvalues.flatten() - fermi_level
        all_energies = all_energies[~np.isnan(all_energies)]
        
        # Set energy window
        if energy_window:
            e_min, e_max = energy_window[0] - fermi_level, energy_window[1] - fermi_level
        else:
            e_min, e_max = np.min(all_energies) - 0.5, np.max(all_energies) + 0.5
        
        # Energy grid for DOS
        energy_grid = np.linspace(e_min, e_max, n_points)
        dos = np.zeros_like(energy_grid)
        
        # Calculate DOS with Gaussian broadening
        for energy in all_energies:
            dos += np.exp(-(energy_grid - energy)**2 / (2 * broadening**2))
        
        dos /= (broadening * np.sqrt(2 * np.pi))  # Normalize
        dos /= len(all_energies) / eigenvalues.shape[1]  # Per unit cell
        
        # Plot DOS
        ax.plot(dos, energy_grid, 'b-', linewidth=2.5, alpha=0.8)
        ax.fill_betweenx(energy_grid, 0, dos, alpha=0.3, color='blue')
        
        # Add Fermi level
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, 
                  alpha=0.8, label='Fermi Level')
        
        # Formatting
        ax.set_xlabel('Density of States (states/eV)', fontsize=14)
        ax.set_ylabel('Energy (eV)', fontsize=14)
        ax.set_title('Electronic Density of States', fontsize=16, pad=15)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎨 Density of states plot saved: {Path(save_path).name}")
        
        return fig, ax
    
    def plot_2d_band_surface(self, eigenvalues, k_points, band_index=0,
                            save_path=None, fermi_level=0.0):
        """
        Plot 2D surface of a specific band in the Brillouin zone.
        
        Args:
            eigenvalues: Band energies, shape (n_k, n_bands)
            k_points: k-point grid, shape (n_k, 2)
            band_index: Which band to plot
            save_path: Path to save figure
            fermi_level: Fermi energy level
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract band energies
        band_energies = eigenvalues[:, band_index] - fermi_level
        
        # Create 3D surface plot
        kx, ky = k_points[:, 0], k_points[:, 1]
        
        # For proper surface plotting, we need to reshape if it's a regular grid
        try:
            # Assume square grid
            n_k = int(np.sqrt(len(kx)))
            if n_k * n_k == len(kx):
                kx_2d = kx.reshape(n_k, n_k)
                ky_2d = ky.reshape(n_k, n_k)
                energies_2d = band_energies.reshape(n_k, n_k)
                
                surf = ax.plot_surface(kx_2d, ky_2d, energies_2d, 
                                     cmap='viridis', alpha=0.8, 
                                     linewidth=0, antialiased=True)
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, 
                           label=f'Band {band_index + 1} Energy (eV)')
            else:
                raise ValueError("Not a regular grid")
                
        except:
            # Fallback to scatter plot
            scatter = ax.scatter(kx, ky, band_energies, 
                               c=band_energies, cmap='viridis', 
                               s=20, alpha=0.7)
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20,
                        label=f'Band {band_index + 1} Energy (eV)')
        
        # Add Fermi level plane
        if fermi_level != 0:
            ax.plot_surface(kx_2d, ky_2d, np.zeros_like(kx_2d), 
                           alpha=0.3, color='red')
        
        # Formatting
        ax.set_xlabel(r'$k_x$', fontsize=14)
        ax.set_ylabel(r'$k_y$', fontsize=14)
        ax.set_zlabel('Energy (eV)', fontsize=14)
        ax.set_title(f'Band {band_index + 1} Surface', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎨 3D band surface saved: {Path(save_path).name}")
        
        return fig, ax

# Convenience functions for direct use
def plot_kitaev_bands(eigenvalues, k_points, save_path=None, **kwargs):
    """
    Quick high-quality plot of Kitaev band structure.
    
    Args:
        eigenvalues: Band energies from analysis
        k_points: k-points from analysis
        save_path: Save path
        **kwargs: Additional plotting parameters
    """
    plotter = DispersionPlotter()
    
    title = 'Kitaev Model: Electronic Band Structure'
    
    return plotter.plot_band_structure(
        eigenvalues, k_points, title=title, 
        save_path=save_path, **kwargs
    )

def plot_gap_analysis(eigenvalues, k_points, save_path=None):
    """
    Plot band structure with detailed gap analysis.
    
    Args:
        eigenvalues: Band energies
        k_points: k-points  
        save_path: Save path
    """
    plotter = DispersionPlotter()
    
    fig, ax = plotter.plot_band_structure(
        eigenvalues, k_points, 
        title='Kitaev Model: Band Gap Analysis',
        show_gap=True, show_fermi=True,
        save_path=save_path
    )
    
    return fig, ax

def plot_diagonal_band_dispersion(eigenvalues, k_points, save_path=None):
    """
    Plot band dispersion along diagonal line from (π,-π) to (-π,π).
    
    This simplified plot shows the band structure along a diagonal path
    that captures all Dirac points in the Kitaev model.
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        save_path: Path to save figure
    """
    # Create square figure for PRL-level quality
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate diagonal path from (π,-π) to (-π,π)
    n_points = 200
    t = np.linspace(0, 1, n_points)
    diagonal_kx = np.pi * (1 - 2*t)  # Range [π, -π]
    diagonal_ky = np.pi * (2*t - 1)  # Range [-π, π]
    diagonal_path = np.column_stack([diagonal_kx, diagonal_ky])
    
    # Interpolate eigenvalues to diagonal path
    from scipy.interpolate import griddata
    
    n_bands = eigenvalues.shape[1]
    interpolated_bands = np.zeros((n_points, n_bands))
    
    for band in range(n_bands):
        interpolated_bands[:, band] = griddata(
            k_points, eigenvalues[:, band], diagonal_path, 
            method='linear', fill_value=np.nan
        )
    
    # Compute exact solution: ±|J(k)|/4 where J(k) = Jz - Jx*exp(ikx) - Jy*exp(iky)
    # Using Jx=Jy=Jz=1.0 for isotropic Kitaev model
    Jx, Jy, Jz = 1.0, 1.0, 1.0
    exact_energies_positive = np.zeros(n_points)
    exact_energies_negative = np.zeros(n_points)
    for i in range(n_points):
        kx = diagonal_kx[i]
        ky = diagonal_ky[i]
        # Compute J(k) = Jz - Jx*exp(ikx) - Jy*exp(iky)
        Jk = Jz - Jx * np.exp(1j * kx) - Jy * np.exp(1j * ky)
        # Eigenvalues are ±|J(k)|/4
        Ek = np.abs(Jk) / 4.0
        exact_energies_positive[i] = Ek
        exact_energies_negative[i] = -Ek
    
    # Plot bands with unified color scheme
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    for band in range(n_bands):
        valid_mask = ~np.isnan(interpolated_bands[:, band])
        if np.any(valid_mask):
            ax.plot(t[valid_mask], interpolated_bands[valid_mask, band], 
                   color=colors[band], linewidth=2.5, alpha=0.9, 
                   label=f'fPEPS Band {band+1}')
    
    # Plot exact solutions (both bands)
    ax.plot(t, exact_energies_positive, color='red', linewidth=3, linestyle='-', 
           alpha=0.8, label='Exact: +|J(k)|/4')
    ax.plot(t, exact_energies_negative, color='red', linewidth=3, linestyle='--', 
           alpha=0.8, label='Exact: -|J(k)|/4')
    
    # Mark only meaningful points: Γ and Dirac points
    meaningful_points = {
        'Γ': 0.5,           # (0, 0) at center
    }
    
    # Add vertical lines and labels for meaningful points
    for name, pos in meaningful_points.items():
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(pos, ax.get_ylim()[1] * 0.95, name, 
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Mark Dirac points at correct positions (1/3 and 2/3)
    dirac_points = {
        'Dirac': 1/3,       # (π/3, -π/3) 
        'Dirac': 2/3,       # (-π/3, π/3)
    }
    
    for i, pos in enumerate([1/3, 2/3]):
        ax.axvline(x=pos, color='red', linestyle='-', linewidth=2, alpha=0.8)
        ax.text(pos, ax.get_ylim()[0] * 0.95, 'Dirac', 
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Add Fermi level
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
              alpha=0.8, label='Fermi Level')
    
    # Professional formatting
    ax.set_xlabel('Momentum', fontsize=16)
    ax.set_ylabel('Energy', fontsize=16)
    ax.set_title('Kitaev Band Dispersion (with Exact Solution)', fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=14, loc='upper right')
    
    # Set x-axis ticks to show only meaningful points
    ax.set_xticks([0, 1/3, 0.5, 2/3, 1])
    ax.set_xticklabels(['(π,-π)', 'Dirac', 'Γ', 'Dirac', '(-π,π)'], fontsize=12)
    
    # Set aspect ratio to square for professional appearance
    ax.set_aspect('auto')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Diagonal band dispersion saved: {Path(save_path).name}")
    
    return fig, ax

def plot_fermi_surface(eigenvalues, k_points, band_index=0, save_path=None):
    """
    Plot Fermi surface for a specific band.
    
    Args:
        eigenvalues: Band energies
        k_points: k-points
        band_index: Which band to analyze
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Find points near Fermi level
    band_energies = eigenvalues[:, band_index]
    fermi_tolerance = 0.05  # eV
    
    fermi_mask = np.abs(band_energies) < fermi_tolerance
    
    if np.any(fermi_mask):
        fermi_k = k_points[fermi_mask]
        ax.scatter(fermi_k[:, 0], fermi_k[:, 1], c='red', s=20, alpha=0.7,
                  label=f'Fermi Surface (Band {band_index + 1})')
    
    # Plot all k-points for reference
    ax.scatter(k_points[:, 0], k_points[:, 1], c='lightgray', s=5, alpha=0.3)
    
    ax.set_xlabel(r'$k_x$', fontsize=14)
    ax.set_ylabel(r'$k_y$', fontsize=14)
    ax.set_title(f'Fermi Surface - Band {band_index + 1}', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Fermi surface plot saved: {Path(save_path).name}")
    
    return fig, ax

def plot_dirac_scaling_analysis(eigenvalues, k_points, dirac_point=(np.pi/3, -np.pi/3), 
                               radius=0.1, save_path=None, n_samples=100):
    """
    Analyze scaling behavior near Dirac point to test quadratic vs linear dispersion.
    
    Uses very small radius and dense sampling for precise analysis near Dirac point.
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        dirac_point: (kx, ky) coordinates of Dirac point
        radius: Radius around Dirac point to analyze (default: 0.1 for dense sampling)
        save_path: Path to save figure
        n_samples: Number of sampling points around Dirac point (default: 100 for dense sampling)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate distances from Dirac point
    distances = np.sqrt((k_points[:, 0] - dirac_point[0])**2 + 
                       (k_points[:, 1] - dirac_point[1])**2)
    
    # Find points within radius of Dirac point
    near_dirac_mask = distances <= radius
    near_dirac_k = k_points[near_dirac_mask]
    near_dirac_eigenvals = eigenvalues[near_dirac_mask]
    near_dirac_distances = distances[near_dirac_mask]
    
    if len(near_dirac_k) < 20:
        print(f"⚠️  Warning: Only {len(near_dirac_k)} points found near Dirac point")
        print(f"   Consider increasing radius (current: {radius}) or k-point density")
        return fig, (ax1, ax2)
    
    # Use all available points for dense analysis (no artificial sampling)
    # Sort by distance for systematic analysis
    sorted_indices = np.argsort(near_dirac_distances)
    sorted_distances = near_dirac_distances[sorted_indices]
    sorted_energies = np.abs(near_dirac_eigenvals[sorted_indices, 0])  # Use first band
    
    # Remove any NaN values
    valid_mask = ~np.isnan(sorted_energies)
    sorted_distances = sorted_distances[valid_mask]
    sorted_energies = sorted_energies[valid_mask]
    
    if len(sorted_energies) < 10:
        print("⚠️  Warning: Insufficient data for scaling analysis")
        return fig, (ax1, ax2)
    
    # Use actual data points for analysis (no artificial sampling)
    sample_dk2 = sorted_distances**2
    sample_energies = sorted_energies
    
    # Plot 1: dK^2 vs E(k)
    ax1.scatter(sample_dk2, sample_energies, c='blue', s=50, alpha=0.7, 
                label='Data points')
    
    # Fit linear relationship: E = α * dK^2
    if len(sample_dk2) > 1:
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
            
        except ImportError:
            print("⚠️  scipy not available for fitting")
    
    ax1.set_xlabel(r'$(\Delta k)^2$', fontsize=14)
    ax1.set_ylabel(r'$|E(k)|$', fontsize=14)
    ax1.set_title(f'Dirac Point Scaling: dK² vs E(k)\nDirac point: ({dirac_point[0]:.2f}, {dirac_point[1]:.2f})', 
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Power law analysis (log-log plot)
    # Use dK instead of dK^2 for power law analysis
    sample_dk = np.sqrt(sample_dk2)
    
    # Remove zero values for log plot
    nonzero_mask = (sample_dk > 0) & (sample_energies > 0)
    if np.sum(nonzero_mask) > 3:
        log_dk = np.log(sample_dk[nonzero_mask])
        log_energy = np.log(sample_energies[nonzero_mask])
        
        ax2.scatter(log_dk, log_energy, c='green', s=50, alpha=0.7, 
                   label='Data points')
        
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
            
        except ImportError:
            print("⚠️  scipy not available for power law fitting")
    
    ax2.set_xlabel(r'$\ln(\Delta k)$', fontsize=14)
    ax2.set_ylabel(r'$\ln(|E(k)|)$', fontsize=14)
    ax2.set_title('Power Law Analysis: ln(dK) vs ln(E)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Dirac scaling analysis saved: {Path(save_path).name}")
    
    return fig, (ax1, ax2)

def plot_2d_band_structure_with_dirac_search(eigenvalues, k_points, save_path=None, 
                                            band_index=0, dirac_search_radius=0.5):
    """
    Plot 2D band structure and mark known Dirac points.
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        save_path: Path to save figure
        band_index: Which band to plot
        dirac_search_radius: Not used anymore, kept for compatibility
    
    Returns:
        fig, ax: Matplotlib figure and axes
        dirac_points: List of known Dirac point coordinates
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reshape data for 2D plotting
    # Assume square grid
    n_k = int(np.sqrt(len(k_points)))
    if n_k * n_k == len(k_points):
        kx = k_points[:, 0].reshape(n_k, n_k)
        ky = k_points[:, 1].reshape(n_k, n_k)
        energies = eigenvalues[:, band_index].reshape(n_k, n_k)
        
        # Create contour plot
        levels = np.linspace(np.min(energies), np.max(energies), 30)
        contour = ax.contourf(kx, ky, energies, levels=levels, 
                             cmap='RdBu_r', alpha=0.8)
        contour_lines = ax.contour(kx, ky, energies, levels=levels, 
                                  colors='black', linewidths=0.5, alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(f'Band {band_index + 1} Energy', fontsize=12)
        
    else:
        # Fallback to scatter plot for irregular grid
        scatter = ax.scatter(k_points[:, 0], k_points[:, 1], 
                           c=eigenvalues[:, band_index], cmap='RdBu_r', 
                           s=20, alpha=0.7)
        fig.colorbar(scatter, ax=ax, shrink=0.8, 
                    label=f'Band {band_index + 1} Energy')
    
    # Mark known Dirac points in [-π, π] system (matching exact_dispersion.py)
    known_dirac_points = [
        (np.pi/3, -np.pi/3),           # Main Dirac point in [-π, π] system
        (-np.pi/3, np.pi/3),           # Equivalent Dirac point
    ]
    
    # Mark known Dirac points
    for i, (kx, ky) in enumerate(known_dirac_points):
        ax.plot(kx, ky, 'ko', markersize=10, markerfacecolor='yellow',
               markeredgewidth=2, markeredgecolor='black')
        if i == 0:  # Only label the main one to avoid clutter
            ax.annotate('Known Dirac\n(π/3, 5π/3)', (kx, ky), xytext=(10, 10), 
                       textcoords='offset points', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Mark high-symmetry points in the correct coordinate system
    
    # Mark high-symmetry points in [-π, π] system (matching exact_dispersion.py)
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
    
    # Formatting
    ax.set_xlabel(r'$k_x$', fontsize=14)
    ax.set_ylabel(r'$k_y$', fontsize=14)
    ax.set_title(f'2D Band Structure with Dirac Point Search\nBand {band_index + 1}\nRange: [0, 2π]', 
                fontsize=16, pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to match exact_dispersion.py [-π, π] system
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 2D band structure with Dirac search saved: {Path(save_path).name}")
    
    return fig, ax, known_dirac_points

def find_dirac_points(eigenvalues, k_points, search_radius=0.5):
    """
    Find Dirac points where bands cross or gap is minimal.
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        search_radius: Radius to search around expected Dirac points
    
    Returns:
        dirac_points: List of (kx, ky) coordinates of found Dirac points
    """
    dirac_points = []
    
    # Method 1: Look for points where bands cross (gap = 0)
    band_gaps = np.abs(eigenvalues[:, 0] - eigenvalues[:, 1])
    min_gap_indices = np.argsort(band_gaps)[:10]  # Top 10 smallest gaps
    
    # Expected Dirac points in the [-π, π] coordinate system
    # Based on exact_dispersion.py, the correct Dirac points are:
    expected_dirac_points = [
        (np.pi/3, -np.pi/3),           # Main Dirac point in [-π, π] system
        (-np.pi/3, np.pi/3),           # Equivalent Dirac point
    ]
    
    for idx in min_gap_indices:
        kx, ky = k_points[idx]
        gap = band_gaps[idx]
        
        # Check if this is near any expected Dirac point
        for expected_dirac in expected_dirac_points:
            distance = np.sqrt((kx - expected_dirac[0])**2 + (ky - expected_dirac[1])**2)
            
            if gap < 0.1 and distance < search_radius:  # Small gap and near expected location
                dirac_points.append((kx, ky))
                print(f"🔍 Found Dirac point at ({kx:.3f}, {ky:.3f}) with gap {gap:.4f}")
                break
    
    # Method 2: Look for points with linear dispersion near expected Dirac points
    for expected_dirac in expected_dirac_points:
        distances = np.sqrt((k_points[:, 0] - expected_dirac[0])**2 + 
                           (k_points[:, 1] - expected_dirac[1])**2)
        
        near_expected = distances < search_radius
        if np.any(near_expected):
            near_k = k_points[near_expected]
            near_eigenvals = eigenvalues[near_expected]
            
            # Check for linear dispersion pattern
            for i, (kx, ky) in enumerate(near_k):
                # Look for linear scaling in nearby points
                local_distances = np.sqrt((near_k[:, 0] - kx)**2 + (near_k[:, 1] - ky)**2)
                local_energies = np.abs(near_eigenvals[:, 0])
                
                # Check if energy scales linearly with distance
                if len(local_distances) > 3:
                    # Simple linearity check
                    sorted_indices = np.argsort(local_distances)
                    sorted_distances = local_distances[sorted_indices]
                    sorted_energies = local_energies[sorted_indices]
                    
                    # Check if first few points show linear scaling
                    if len(sorted_distances) >= 4:
                        # Calculate correlation for first 4 points
                        x = sorted_distances[:4]
                        y = sorted_energies[:4]
                        
                        if len(x) > 1:
                            correlation = np.corrcoef(x, y)[0, 1]
                            if correlation > 0.8:  # Good linear correlation
                                dirac_points.append((kx, ky))
                                print(f"🔍 Found potential Dirac point at ({kx:.3f}, {ky:.3f}) with linear correlation {correlation:.3f}")
    
    # Remove duplicates
    unique_dirac_points = []
    for point in dirac_points:
        is_duplicate = False
        for existing in unique_dirac_points:
            distance = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
            if distance < 0.1:  # Within 0.1 units
                is_duplicate = True
                break
        if not is_duplicate:
            unique_dirac_points.append(point)
    
    return unique_dirac_points

def plot_dirac_scaling_analysis_auto(eigenvalues, k_points, save_path=None, 
                                    search_radius=0.5, analysis_radius=0.3):
    """
    Automatically find Dirac points and perform scaling analysis.
    
    Args:
        eigenvalues: Band energies, shape (n_k, n_bands)
        k_points: k-point grid, shape (n_k, 2)
        save_path: Path to save figure
        search_radius: Radius to search for Dirac points
        analysis_radius: Radius for scaling analysis around found Dirac points
    
    Returns:
        fig, ax: Matplotlib figure and axes
        dirac_points: List of found Dirac point coordinates
    """
    # First, find Dirac points
    dirac_points = find_dirac_points(eigenvalues, k_points, search_radius)
    
    if not dirac_points:
        print("⚠️  No Dirac points found automatically")
        print("   Trying with expected Dirac point (π/3, -π/3)")
        dirac_points = [(np.pi/3, -np.pi/3)]
    
    # Create subplots for each found Dirac point
    n_dirac = len(dirac_points)
    if n_dirac == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        axes = [(ax1, ax2)]
    else:
        fig, axes = plt.subplots(n_dirac, 2, figsize=(15, 6*n_dirac))
        if n_dirac == 1:
            axes = [axes]
    
    for i, dirac_point in enumerate(dirac_points):
        if n_dirac == 1:
            ax1, ax2 = axes[0]
        else:
            ax1, ax2 = axes[i]
        
        print(f"🔬 Analyzing Dirac point {i+1}: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})")
        
        # Perform scaling analysis directly on these axes
        # Calculate distances from Dirac point
        distances = np.sqrt((k_points[:, 0] - dirac_point[0])**2 + 
                           (k_points[:, 1] - dirac_point[1])**2)
        
        # Find points within analysis radius
        near_dirac_mask = distances <= analysis_radius
        near_dirac_k = k_points[near_dirac_mask]
        near_dirac_eigenvals = eigenvalues[near_dirac_mask]
        near_dirac_distances = distances[near_dirac_mask]
        
        if len(near_dirac_k) < 10:
            print(f"⚠️  Warning: Only {len(near_dirac_k)} points found near Dirac point")
            continue
        
        # Sample points at different distances for systematic analysis
        max_dist = np.max(near_dirac_distances)
        sample_distances = np.linspace(0.01, max_dist, 50)
        
        # For each sample distance, find the average energy
        sample_energies = []
        sample_dk2 = []  # dK^2 values
        
        for d in sample_distances:
            # Find points at approximately this distance
            tolerance = max_dist / 50
            mask = np.abs(near_dirac_distances - d) < tolerance
            
            if np.any(mask):
                # Average energy at this distance
                energies_at_d = near_dirac_eigenvals[mask]
                avg_energy = np.mean(np.abs(energies_at_d))  # Use absolute value
                sample_energies.append(avg_energy)
                sample_dk2.append(d**2)
        
        sample_energies = np.array(sample_energies)
        sample_dk2 = np.array(sample_dk2)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(sample_energies) | np.isnan(sample_dk2))
        sample_energies = sample_energies[valid_mask]
        sample_dk2 = sample_dk2[valid_mask]
        
        if len(sample_energies) < 5:
            print("⚠️  Warning: Insufficient data for scaling analysis")
            continue
        
        # Plot 1: dK^2 vs E(k)
        ax1.scatter(sample_dk2, sample_energies, c='blue', s=50, alpha=0.7, 
                    label='Data points')
        
        # Fit linear relationship: E = α * dK^2
        if len(sample_dk2) > 1:
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
                
            except ImportError:
                print("⚠️  scipy not available for fitting")
        
        ax1.set_xlabel(r'$(\Delta k)^2$', fontsize=14)
        ax1.set_ylabel(r'$|E(k)|$', fontsize=14)
        ax1.set_title(f'Dirac Point {i+1}: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})\ndK² vs E(k)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Plot 2: Power law analysis (log-log plot)
        # Use dK instead of dK^2 for power law analysis
        sample_dk = np.sqrt(sample_dk2)
        
        # Remove zero values for log plot
        nonzero_mask = (sample_dk > 0) & (sample_energies > 0)
        if np.sum(nonzero_mask) > 3:
            log_dk = np.log(sample_dk[nonzero_mask])
            log_energy = np.log(sample_energies[nonzero_mask])
            
            ax2.scatter(log_dk, log_energy, c='green', s=50, alpha=0.7, 
                       label='Data points')
            
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
                
            except ImportError:
                print("⚠️  scipy not available for power law fitting")
        
        ax2.set_xlabel(r'$\ln(\Delta k)$', fontsize=14)
        ax2.set_ylabel(r'$\ln(|E(k)|)$', fontsize=14)
        ax2.set_title(f'Dirac Point {i+1}: ({dirac_point[0]:.3f}, {dirac_point[1]:.3f})\nPower Law Analysis', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Auto Dirac scaling analysis saved: {Path(save_path).name}")
    
    return fig, axes, dirac_points

if __name__ == "__main__":
    # Example usage and testing
    print("🎨 High-quality band dispersion plotter ready!")
    print("📊 Features:")
    print("  - PRL-level publication quality")
    print("  - High-symmetry path interpolation")
    print("  - Band gap analysis")
    print("  - Density of states")
    print("  - 3D band surfaces") 
    print("  - Fermi surface analysis")
