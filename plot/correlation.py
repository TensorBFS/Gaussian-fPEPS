#!/usr/bin/env python3
"""
High-quality scientific plotting for real-space correlation functions G(r).
PRL-level publication-ready figures for computational physics.

Author: Kitaev-fPEPS Project
Style: Physical Review Letters computational physics standards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# Custom colormaps for physics
def create_physics_colormap():
    """Create custom colormap suitable for correlation functions."""
    colors_list = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    return colors.LinearSegmentedColormap.from_list('physics', colors_list)

def create_diverging_colormap():
    """Create diverging colormap for correlation functions that can be positive/negative."""
    return plt.cm.RdBu_r

# Complex correlation plotting functions removed - simplified version only keeps essential diagonal plotting

def plot_correlation_scaling_analysis_diagonal(G_r, r_points, component=(0,0), 
                                             save_path=None, max_distance=None):
    """
    Plot correlation function scaling behavior along diagonal (x+y) direction.
    
    Args:
        G_r: Real-space correlation function
        r_points: Coordinate grid
        component: Which component to analyze
        save_path: Save path for the plot
        max_distance: Maximum distance to include in fit
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract correlation component
    G_component = G_r[:, :, component[0], component[1]]
    
    # Get center indices
    center_x = G_r.shape[0] // 2
    center_y = G_r.shape[1] // 2
    
    # Extract diagonal profile (x+y direction)
    # We'll sample along the diagonal from center
    max_diag_len = min(G_r.shape[0], G_r.shape[1])
    diag_profile = []
    diag_distances = []
    
    # Sample along positive diagonal (x+y direction)
    for i in range(max_diag_len):
        x_idx = center_x + i
        y_idx = center_y + i
        
        # Check bounds
        if x_idx < G_r.shape[0] and y_idx < G_r.shape[1]:
            diag_profile.append(G_component[x_idx, y_idx])
            # Calculate distance along diagonal: r = sqrt(2) * i
            diag_distances.append(i * np.sqrt(2))
    
    # Sample along negative diagonal (x-y direction) for comparison
    diag_profile_neg = []
    diag_distances_neg = []
    
    for i in range(max_diag_len):
        x_idx = center_x + i
        y_idx = center_y - i
        
        # Check bounds
        if x_idx < G_r.shape[0] and y_idx >= 0:
            diag_profile_neg.append(G_component[x_idx, y_idx])
            diag_distances_neg.append(i * np.sqrt(2))
    
    # Convert to numpy arrays
    diag_profile = np.array(diag_profile)
    diag_distances = np.array(diag_distances)
    diag_profile_neg = np.array(diag_profile_neg)
    diag_distances_neg = np.array(diag_distances_neg)
    
    # Combine positive and negative directions
    all_distances = np.concatenate([diag_distances])
    all_profile = np.concatenate([diag_profile])
    
    # Remove center point and sort by distance
    valid_mask = all_distances > 0
    distances = all_distances[valid_mask]
    correlation_values = np.abs(all_profile[valid_mask])
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    correlation_values = correlation_values[sort_idx]
    
    # Apply max_distance filter if specified
    if max_distance:
        mask = distances <= max_distance
        distances = distances[mask]
        correlation_values = correlation_values[mask]
    
    # Remove zeros and very small values for log plot
    nonzero_mask = correlation_values > 1e-10
    distances_fit = distances[nonzero_mask]
    corr_fit = correlation_values[nonzero_mask]
    
    # Linear plot
    ax1.plot(distances, correlation_values, 'o-', markersize=6, linewidth=2, 
            alpha=0.8, label=f'$|G_{{{component[0]}{component[1]}}}(r)|$')
    ax1.set_xlabel(r'Distance $r_{x+y}$ (lattice units)', fontsize=14)
    ax1.set_ylabel(f'$|G_{{{component[0]}{component[1]}}}(r)|$', fontsize=14)
    ax1.set_title(f'Correlation Decay (Diagonal) - Component ({component[0]},{component[1]})', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Log-log plot with power law fitting
    if len(distances_fit) > 3:  # Need at least 3 points for fitting
        # Power law fitting: |G(r)| = A * r^(-alpha)
        # log(|G(r)|) = log(A) - alpha * log(r)
        log_r = np.log(distances_fit)
        log_G = np.log(corr_fit)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_r, log_G, 1)
        alpha = -coeffs[0]  # Power law exponent
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Generate fitted curve
        r_fit = np.linspace(distances_fit.min(), distances_fit.max(), 100)
        G_fit = A * r_fit**(-alpha)
        
        # Plot data and fit
        ax2.loglog(distances_fit, corr_fit, 'o', markersize=8, alpha=0.8, 
                  label=f'$|G_{{{component[0]}{component[1]}}}(r)|$')
        ax2.loglog(r_fit, G_fit, '--', linewidth=3, alpha=0.9,
                  label=f'Power Law: $r^{{-{alpha:.2f}}}$')
        
        # Add power law equation as text
        ax2.text(0.05, 0.95, f'$|G(r)| \\propto r^{{-{alpha:.2f}}}$', 
                transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        print(f"🔬 Power law fit for G_{{{component[0]}{component[1]}}} (diagonal): α = {alpha:.3f}")
        
    else:
        ax2.loglog(distances, correlation_values, 'o-', markersize=6, 
                  label=f'$|G_{{{component[0]}{component[1]}}}(r)|$')
        ax2.text(0.05, 0.95, 'Insufficient data for fitting', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel(r'Distance $r_{x+y}$ (lattice units)', fontsize=14)
    ax2.set_ylabel(f'$|G_{{{component[0]}{component[1]}}}(r)|$', fontsize=14)
    ax2.set_title(f'Power Law Analysis (Diagonal) - Component ({component[0]},{component[1]})', fontsize=16)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Diagonal correlation scaling analysis saved: {Path(save_path).name}")
    
    return fig, (ax1, ax2)

if __name__ == "__main__":
    # Simplified correlation plotter - only diagonal scaling analysis
    print("🎨 Simplified correlation function plotter ready!")
    print("📊 Features:")
    print("  - Diagonal power law scaling analysis only")
    print("  - Other functions moved to analysis_correlation.py")
