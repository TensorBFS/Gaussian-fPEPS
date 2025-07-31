"""
High-quality scientific plotting package for Kitaev-fPEPS.
PRL-level publication-ready figures for computational physics.
"""

try:
    from .correlation import (
        CorrelationPlotter, 
        plot_kitaev_correlation, 
        plot_correlation_decay
    )
    CORRELATION_AVAILABLE = True
except ImportError:
    CORRELATION_AVAILABLE = False

try:
    from .dispersion import (
        DispersionPlotter, 
        plot_kitaev_bands, 
        plot_gap_analysis, 
        plot_fermi_surface
    )
    DISPERSION_AVAILABLE = True
except ImportError:
    DISPERSION_AVAILABLE = False

__all__ = []

if CORRELATION_AVAILABLE:
    __all__.extend([
        'CorrelationPlotter', 
        'plot_kitaev_correlation', 
        'plot_correlation_decay'
    ])

if DISPERSION_AVAILABLE:
    __all__.extend([
        'DispersionPlotter', 
        'plot_kitaev_bands', 
        'plot_gap_analysis', 
        'plot_fermi_surface'
    ])

# Module information
__version__ = "1.0.0"
__author__ = "Kitaev-fPEPS Project"
__description__ = "PRL-level scientific plotting for computational physics" 