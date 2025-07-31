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

class CorrelationPlotter:
    """High-quality plotter for real-space correlation functions."""
    
    def __init__(self, figsize=(10, 8), style='physics'):
        """
        Initialize the correlation plotter.
        
        Args:
            figsize: Figure size in inches
            style: 'physics' or 'diverging' colormap style
        """
        self.figsize = figsize
        self.style = style
        self.cmap = create_physics_colormap() if style == 'physics' else create_diverging_colormap()
    
    def plot_correlation_heatmap(self, G_r, r_points, component=(0,0), 
                                title=None, save_path=None, 
                                vmin=None, vmax=None, 
                                show_lattice=True, interpolation='bilinear'):
        """
        Plot high-quality heatmap of real-space correlation function.
        
        Args:
            G_r: Real-space correlation function, shape (rx, ry, i, j)
            r_points: Coordinate grid, shape (rx, ry, 2)
            component: Which component (i,j) to plot
            title: Custom title for the plot
            save_path: Path to save the figure
            vmin, vmax: Color scale limits
            show_lattice: Whether to overlay lattice sites
            interpolation: Interpolation method for smooth visualization
        """
        # Extract the correlation component
        G_component = G_r[:, :, component[0], component[1]]
        
        # Handle complex correlations by taking real part and magnitude
        if np.iscomplexobj(G_component):
            G_real = np.real(G_component)
            G_imag = np.imag(G_component)
            G_magnitude = np.abs(G_component)
            plot_complex = True
        else:
            G_real = G_component
            plot_complex = False
        
        # Create figure with subplots for complex data
        if plot_complex:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            data_list = [G_real, G_imag, G_magnitude]
            titles = ['Real Part', 'Imaginary Part', 'Magnitude']
            cmaps = [self.cmap, self.cmap, create_physics_colormap()]
        else:
            fig, axes = plt.subplots(1, 1, figsize=self.figsize)
            axes = [axes]
            data_list = [G_real]
            titles = ['']
            cmaps = [self.cmap]
        
        # Extract coordinate information
        rx_coords = r_points[:, 0, 0]
        ry_coords = r_points[0, :, 1]
        extent = [rx_coords[0], rx_coords[-1], ry_coords[0], ry_coords[-1]]
        
        for ax, data, subtitle, cmap in zip(axes, data_list, titles, cmaps):
            # Set color scale
            if vmin is None or vmax is None:
                vmin_local = np.min(data)
                vmax_local = np.max(data)
                # Make symmetric for diverging data
                if self.style == 'diverging':
                    vmax_abs = max(abs(vmin_local), abs(vmax_local))
                    vmin_local, vmax_local = -vmax_abs, vmax_abs
            else:
                vmin_local, vmax_local = vmin, vmax
            
            # Create main heatmap
            im = ax.imshow(data.T, origin='lower', cmap=cmap, 
                          extent=extent, aspect='equal',
                          vmin=vmin_local, vmax=vmax_local,
                          interpolation=interpolation)
            
            # Add contour lines for better visualization
            if data.shape[0] > 5 and data.shape[1] > 5:  # Only if sufficient resolution
                # Ensure X, Y meshgrid matches data dimensions
                if len(rx_coords) == data.shape[0] and len(ry_coords) == data.shape[1]:
                    X, Y = np.meshgrid(rx_coords, ry_coords, indexing='ij')
                    contours = ax.contour(X, Y, data, levels=8, colors='white', 
                                        alpha=0.3, linewidths=0.8)
                else:
                    # Skip contours if dimensions don't match
                    pass
            
            # Overlay lattice sites if requested (reduce density for better visibility)
            if show_lattice:
                # Only show every other lattice site to reduce clutter
                step = max(1, len(rx_coords) // 15)  # Adaptive step size
                for i in range(0, len(rx_coords), step):
                    for j in range(0, len(ry_coords), step):
                        ax.plot(rx_coords[i], ry_coords[j], 'o', 
                               color='white', markersize=2, alpha=0.6,
                               markeredgecolor='black', markeredgewidth=0.3)
            
            # Formatting
            ax.set_xlabel(r'$r_x$ (lattice units)', fontsize=14)
            ax.set_ylabel(r'$r_y$ (lattice units)', fontsize=14)
            
            if title and not plot_complex:
                ax.set_title(title, fontsize=16, pad=15)
            elif plot_complex:
                ax.set_title(f'{subtitle}', fontsize=14, pad=10)
            
            # Set integer ticks for lattice coordinates
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(f'$G_{{{component[0]}{component[1]}}}(r)$', 
                          fontsize=14, rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=12)
            
            # Grid styling
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
        
        # Overall title for complex plots
        if plot_complex and title:
            fig.suptitle(f'{title} - Component ({component[0]},{component[1]})', 
                        fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"🎨 High-quality correlation plot saved: {Path(save_path).name}")
        
        return fig, axes
    
    def plot_correlation_profile(self, G_r, r_points, component=(0,0), 
                               direction='x', center_y=0, save_path=None):
        """
        Plot 1D profile of correlation function along specified direction.
        
        Args:
            G_r: Real-space correlation function
            r_points: Coordinate grid
            component: Which component to plot
            direction: 'x' or 'y' direction
            center_y: Center position for x-direction cut (or center_x for y-direction)
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract correlation component
        G_component = G_r[:, :, component[0], component[1]]
        
        if direction == 'x':
            # Find closest y index to center_y
            ry_coords = r_points[0, :, 1]
            center_idx = np.argmin(np.abs(ry_coords - center_y))
            
            x_coords = r_points[:, center_idx, 0]
            correlation_profile = G_component[:, center_idx]
            xlabel = r'$r_x$ (lattice units)'
            profile_label = f'$r_y = {ry_coords[center_idx]:.1f}$'
        else:
            # Find closest x index to center_x (using center_y parameter)
            rx_coords = r_points[:, 0, 0]
            center_idx = np.argmin(np.abs(rx_coords - center_y))
            
            y_coords = r_points[center_idx, :, 1]
            correlation_profile = G_component[center_idx, :]
            xlabel = r'$r_y$ (lattice units)'
            profile_label = f'$r_x = {rx_coords[center_idx]:.1f}$'
        
        # Handle complex data
        if np.iscomplexobj(correlation_profile):
            coords = x_coords if direction == 'x' else y_coords
            ax.plot(coords, np.real(correlation_profile), 'o-', 
                   label=f'Re[$G_{{{component[0]}{component[1]}}}$], {profile_label}',
                   linewidth=2, markersize=6, alpha=0.8)
            ax.plot(coords, np.imag(correlation_profile), 's-', 
                   label=f'Im[$G_{{{component[0]}{component[1]}}}$], {profile_label}',
                   linewidth=2, markersize=6, alpha=0.8)
            ax.plot(coords, np.abs(correlation_profile), '^-', 
                   label=f'$|G_{{{component[0]}{component[1]}}}|$, {profile_label}',
                   linewidth=2, markersize=6, alpha=0.8)
        else:
            coords = x_coords if direction == 'x' else y_coords
            ax.plot(coords, correlation_profile, 'o-', 
                   label=f'$G_{{{component[0]}{component[1]}}}$, {profile_label}',
                   linewidth=2.5, markersize=8, alpha=0.9)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(f'$G_{{{component[0]}{component[1]}}}(r)$', fontsize=14)
        ax.set_title(f'Correlation Function Profile - Component ({component[0]},{component[1]})', 
                    fontsize=16, pad=15)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12, loc='best')
        
        # Axis formatting
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎨 Correlation profile plot saved: {Path(save_path).name}")
        
        return fig, ax
    
    def plot_all_components(self, G_r, r_points, save_dir=None, prefix="correlation"):
        """
        Plot all components of the correlation matrix.
        
        Args:
            G_r: Real-space correlation function, shape (rx, ry, i, j)
            r_points: Coordinate grid
            save_dir: Directory to save plots
            prefix: Filename prefix
        """
        n_components = G_r.shape[2]
        
        for i in range(n_components):
            for j in range(n_components):
                title = f'Correlation Function $G_{{{i}{j}}}(r)$'
                
                if save_dir:
                    save_path = Path(save_dir) / f"{prefix}_component_{i}{j}.png"
                else:
                    save_path = None
                
                fig, ax = self.plot_correlation_heatmap(
                    G_r, r_points, component=(i,j), 
                    title=title, save_path=save_path
                )
                plt.close(fig)  # Close to save memory
        
        print(f"🎨 All correlation components plotted and saved to {save_dir}")

# Convenience functions for direct use
def plot_kitaev_correlation(G_r, r_points, component=(0,0), 
                          style='diverging', save_path=None, **kwargs):
    """
    Quick high-quality plot of Kitaev correlation function.
    
    Args:
        G_r: Real-space correlation function
        r_points: Coordinate grid  
        component: Component to plot
        style: 'physics' or 'diverging'
        save_path: Save path
        **kwargs: Additional plotting parameters
    """
    plotter = CorrelationPlotter(style=style)
    
    title = f'Kitaev Model: Real-Space Correlation $G_{{{component[0]}{component[1]}}}(r)$'
    
    return plotter.plot_correlation_heatmap(
        G_r, r_points, component=component, 
        title=title, save_path=save_path, **kwargs
    )

def plot_correlation_decay(G_r, r_points, component=(0,0), save_path=None):
    """
    Plot correlation function decay with distance.
    
    Args:
        G_r: Real-space correlation function
        r_points: Coordinate grid
        component: Component to plot
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate distances from center
    center_x = G_r.shape[0] // 2
    center_y = G_r.shape[1] // 2
    
    distances = []
    correlations = []
    
    G_component = G_r[:, :, component[0], component[1]]
    
    for i in range(G_r.shape[0]):
        for j in range(G_r.shape[1]):
            dx = r_points[i, j, 0] - r_points[center_x, center_y, 0]
            dy = r_points[i, j, 1] - r_points[center_x, center_y, 1]
            r = np.sqrt(dx**2 + dy**2)
            
            if r > 0:  # Skip center point
                distances.append(r)
                correlations.append(np.abs(G_component[i, j]))
    
    # Sort by distance for smooth curve
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    correlations = np.array(correlations)[sorted_indices]
    
    # Plot
    ax.semilogy(distances, correlations, 'o', alpha=0.7, markersize=6,
               label=f'$|G_{{{component[0]}{component[1]}}}(r)|$')
    
    # Formatting
    ax.set_xlabel(r'Distance $r$ (lattice units)', fontsize=14)
    ax.set_ylabel(f'$|G_{{{component[0]}{component[1]}}}(r)|$', fontsize=14)
    ax.set_title('Correlation Function Decay', fontsize=16, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Correlation decay plot saved: {Path(save_path).name}")
    
    return fig, ax

# Power law fitting and scaling analysis
def plot_correlation_scaling_analysis(G_r, r_points, component=(0,0), direction='x', 
                                     save_path=None, max_distance=None):
    """
    Plot correlation function scaling behavior with power law fitting.
    
    Args:
        G_r: Real-space correlation function
        r_points: Coordinate grid
        component: Which component to analyze
        direction: 'x' or 'y' direction for the cut
        save_path: Save path for the plot
        max_distance: Maximum distance to include in fit
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract correlation component
    G_component = G_r[:, :, component[0], component[1]]
    
    # Get center indices
    center_x = G_r.shape[0] // 2
    center_y = G_r.shape[1] // 2
    
    if direction == 'x':
        # Extract x-direction profile through center
        profile = G_component[:, center_y]
        coords = r_points[:, center_y, 0]
        center_idx = center_x
        xlabel = r'Distance $r_x$ (lattice units)'
    else:
        # Extract y-direction profile through center
        profile = G_component[center_x, :]
        coords = r_points[center_x, :, 1]
        center_idx = center_y
        xlabel = r'Distance $r_y$ (lattice units)'
    
    # Ensure profile and coords have same length
    min_len = min(len(profile), len(coords))
    profile = profile[:min_len]
    coords = coords[:min_len]
    
    # Get distances from center
    if center_idx >= len(coords):
        center_idx = len(coords) // 2
    
    distances = np.abs(coords - coords[center_idx])
    correlation_values = np.abs(profile)
    
    # Remove center point and sort by distance
    valid_mask = distances > 0
    distances = distances[valid_mask]
    correlation_values = correlation_values[valid_mask]
    
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
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(f'$|G_{{{component[0]}{component[1]}}}(r)|$', fontsize=14)
    ax1.set_title(f'Correlation Decay - Component ({component[0]},{component[1]})', fontsize=16)
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
        
        print(f"🔬 Power law fit for G_{{{component[0]}{component[1]}}}: α = {alpha:.3f}")
        
    else:
        ax2.loglog(distances, correlation_values, 'o-', markersize=6, 
                  label=f'$|G_{{{component[0]}{component[1]}}}(r)|$')
        ax2.text(0.05, 0.95, 'Insufficient data for fitting', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel(f'$|G_{{{component[0]}{component[1]}}}(r)|$', fontsize=14)
    ax2.set_title(f'Power Law Analysis - Component ({component[0]},{component[1]})', fontsize=16)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Correlation scaling analysis saved: {Path(save_path).name}")
    
    return fig, (ax1, ax2)

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

def plot_correlation_4fold_symmetry(G_r, r_points, component=(0,0), save_path=None):
    """
    Plot correlation function with 4-fold tiling to center the peak.
    
    Args:
        G_r: Real-space correlation function
        r_points: Coordinate grid
        component: Which component to plot
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract correlation component
    G_component = G_r[:, :, component[0], component[1]]
    
    # Create 4-fold tiled version
    G_tiled = np.block([[G_component, G_component],
                        [G_component, G_component]])
    
    # Create corresponding coordinate system
    nx, ny = G_component.shape
    rx_orig = r_points[:, 0, 0]
    ry_orig = r_points[0, :, 1]
    
    # Extend coordinates for tiled version
    rx_min, rx_max = rx_orig.min(), rx_orig.max()
    ry_min, ry_max = ry_orig.min(), ry_orig.max()
    
    rx_extended = np.concatenate([rx_orig, rx_orig + (rx_max - rx_min)])
    ry_extended = np.concatenate([ry_orig, ry_orig + (ry_max - ry_min)])
    
    extent = [rx_extended[0], rx_extended[-1], ry_extended[0], ry_extended[-1]]
    
    # Plot magnitude
    if np.iscomplexobj(G_tiled):
        data_to_plot = np.abs(G_tiled)
        title_suffix = "Magnitude"
    else:
        data_to_plot = G_tiled
        title_suffix = ""
    
    # Color scale
    vmax = np.max(np.abs(data_to_plot))
    vmin = -vmax if not np.iscomplexobj(G_tiled) else 0
    
    im = ax.imshow(data_to_plot.T, origin='lower', cmap='RdBu_r', 
                   extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    
    # Mark the center (where peak should be)
    center_x = (rx_extended[0] + rx_extended[-1]) / 2
    center_y = (ry_extended[0] + ry_extended[-1]) / 2
    ax.plot(center_x, center_y, '+', color='yellow', markersize=15, markeredgewidth=3)
    
    # Formatting
    ax.set_xlabel(r'$r_x$ (lattice units)', fontsize=14)
    ax.set_ylabel(r'$r_y$ (lattice units)', fontsize=14)
    ax.set_title(f'4-Fold Tiled Correlation $G_{{{component[0]}{component[1]}}}(r)$ {title_suffix}', 
                fontsize=16, pad=15)
    
    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(f'$G_{{{component[0]}{component[1]}}}(r)$ {title_suffix}', 
                  fontsize=14, rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎨 4-fold correlation plot saved: {Path(save_path).name}")
    
    return fig, ax

if __name__ == "__main__":
    # Example usage and testing
    print("🎨 Enhanced correlation function plotter ready!")
    print("📊 New features:")
    print("  - Power law scaling analysis")
    print("  - 4-fold symmetry visualization") 
    print("  - Reduced lattice site density")
    print("  - Enhanced fitting capabilities")
