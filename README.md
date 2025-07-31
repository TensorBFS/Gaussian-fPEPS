# Kitaev-fPEPS

**Kitaev honeycomb spin-1/2 model using Gaussian fermionic Projected Entangled Pair States (fPEPS)**

A highly optimized, single-purpose implementation for studying the Kitaev honeycomb model with Majorana fermions.

## 🎯 About

This is a streamlined implementation focusing exclusively on the **Kitaev honeycomb spin-1/2 model** using Gaussian fPEPS. The code is designed for maximum simplicity, clarity, and computational efficiency.

### Key Features
- ✅ **Single-purpose**: Kitaev honeycomb model only  
- ✅ **All-in-one**: Complete implementation in `src/kitaev.py`
- ✅ **Zero ambiguity**: Fixed parameters (Nf=1), no multi-model confusion
- ✅ **High performance**: JAX-based with JIT compilation
- ✅ **Majorana representation**: Direct implementation of fermionic PEPS

## 🔬 Physics

The Kitaev honeycomb model in Majorana representation:

```
H = i ∑_{⟨jk⟩} (Jₓγⱼˣγₖˣ + Jᵧγⱼᵧγₖᵧ + Jᵢγⱼᵢγₖᵢ)
```

Ground state energies (benchmark):
- D=2: E = -0.1906273
- D=4: E = -0.1967948
- D=8: E = -0.1968234  
- D=∞: E = -0.196824657567299

## 🚀 Usage

### Optimization
```bash
# Use defaults
python optimize.py

# Override specific parameters  
python optimize.py --Jx 0.5 --Jy 1.0 --Jz 1.0 --Lx 100 --Ly 100 --Nv 3

# Quick debug run
python optimize.py --Lx 5 --Ly 5 --Nv 1 --max_iterations 10
```

### Analysis
```bash
# Analyze saved results with high-resolution k-space (auto-detects seed directory)
python analysis.py results/seed42/kitaev_*_Glocal.npy --Lx 200 --Ly 200

# Or use metadata file (results saved to same seed directory)
python analysis.py results/seed123/kitaev_*_meta.json --Lx 150 --Ly 150 --max_distance 15

# Override output directory if needed
python analysis.py results/seed42/kitaev_*_Glocal.npy --no_plots --output_dir custom_analysis
```

### Configuration-based Interface
```python
from kitaev import run_kitaev_simulation

# Use defaults
kitaev_system, result = run_kitaev_simulation()

# Override specific parameters
config = {
    "Lx": 100, "Ly": 100, "Nv": 3,
    "Jx": 0.5, "max_iterations": 500
}
kitaev_system, result = run_kitaev_simulation(config)

print(f"Ground state energy: {kitaev_system.energy():.8f}")
```

### Direct Class Usage
```python
from kitaev import KitaevfPEPS, optimize

# Create system
kitaev = KitaevfPEPS(Lx=50, Ly=50, Nv=2, Jx=1.0, Jy=1.0, Jz=1.0)

# Optimize
optimized_kitaev, result = optimize(kitaev, max_iterations=1000)
```

### Complete Workflow: Optimization → Analysis

**Step 1: Optimization**
```bash
python optimize.py --Jx 1.0 --Jy 1.0 --Jz 1.0 --Nv 3 --max_iterations 1000
```

**Step 2: Analysis**
```bash
python analysis.py results/seed42/kitaev_*_Glocal.npy --Lx 200 --Ly 200
```

**Data and Visualization Files Generated:**
```
results/
├── seed42/                                  # All files for seed=42
│   ├── kitaev_Jx1.0_Jy1.0_Jz1.0_Nv3_*_T.npy          # T matrix
│   ├── kitaev_Jx1.0_Jy1.0_Jz1.0_Nv3_*_Glocal.npy     # Glocal = T^T @ J @ T  
│   ├── kitaev_Jx1.0_Jy1.0_Jz1.0_Nv3_*_meta.json      # Optimization metadata
│   ├── analysis_eigenvalues.npy            # Band eigenvalues
│   ├── analysis_G_r.npy                    # Real-space correlations  
│   ├── analysis_metadata.json              # Analysis parameters
│   ├── 🎨 PRL-Quality Visualizations:
│   ├── band_dispersion_hq.png              # 🎵 High-quality band structure  
│   ├── correlation_G00_hq.png              # 🌐 G₀₀(r) correlation heatmap
│   ├── correlation_G01_hq.png              # 🌐 G₀₁(r) correlation heatmap
│   ├── correlation_G10_hq.png              # 🌐 G₁₀(r) correlation heatmap
│   ├── correlation_G11_hq.png              # 🌐 G₁₁(r) correlation heatmap
│   ├── density_of_states.png               # 📈 Electronic DOS
│   ├── correlation_decay.png               # 📉 Correlation vs distance
│   └── band_gap_analysis.png               # 🔍 Gap analysis with annotation
├── seed123/                                 # All files for seed=123
│   └── ... (same structure with different parameters)
└── seed456/
    └── ... (more seeds with complete analysis)
```

**Programmatic Analysis:**
```python
from analysis import run_analysis

# Run complete analysis pipeline (results auto-saved to seed directory)
results = run_analysis(
    glocal_file="results/seed42/kitaev_*_Glocal.npy",
    Lx=200, Ly=200,           # High-resolution k-space
    max_distance=15,          # Extended real-space range
    save_plots=True           # PRL-quality plots auto-generated
)

# Access results
eigenvalues = results['eigenvalues']  # Band structure
G_r = results['G_r']                  # Real-space correlations

# Direct high-quality plotting
from plot.dispersion import plot_kitaev_bands
from plot.correlation import plot_kitaev_correlation

# Publication-ready band structure
plot_kitaev_bands(eigenvalues, k_points, save_path="custom_bands.png")

# Publication-ready correlation function
plot_kitaev_correlation(G_r, r_points, component=(0,0), 
                       save_path="custom_correlation.png")
```

## 🏆 Scientific Excellence

This implementation delivers **publication-ready results** with:
- **Theoretical rigor**: Full Majorana representation
- **Numerical precision**: JAX double-precision computation  
- **Computational efficiency**: JIT-compiled optimization
- **Visualization quality**: PRL-standard scientific plots
- **Reproducibility**: Seed-based file organization
- **Extensibility**: Modular, well-documented code

## 📦 Installation

### Dependencies
```bash
pip install jax jaxlib pymanopt numpy matplotlib scipy seaborn
```

## 🎨 PRL-Level Scientific Plotting

This project includes **publication-ready, Physical Review Letters quality** plotting capabilities:

### Features
- **Computer Modern Roman** fonts (LaTeX standard)
- **300 DPI** resolution for publication
- **High-symmetry path** interpolation (Γ-M-K-Γ for honeycomb lattice)
- **Automatic band gap** detection and annotation
- **Complex correlation functions** (real, imaginary, magnitude)
- **Lattice site overlays** and contour lines
- **Scientific color schemes** optimized for physics

### Generated Plots
1. **Band Structure** (`band_dispersion_hq.png`)
   - High-symmetry path interpolation
   - Band gap analysis and annotation
   - Professional axis formatting

2. **Correlation Functions** (`correlation_G{i}{j}_hq.png`)
   - Real-space heatmaps with lattice overlay
   - Complex data support (Re/Im/|·|)
   - Multiple component visualization

3. **Density of States** (`density_of_states.png`)
   - Gaussian broadening
   - Fermi level marking

4. **Correlation Decay** (`correlation_decay.png`)
   - Distance-dependent analysis
   - Semi-log plotting

5. **Band Gap Analysis** (`band_gap_analysis.png`)
   - Automatic gap detection
   - Valence/conduction band identification

### For GPU support:
```bash
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Development
```bash
pip install pytest  # for running tests
pytest tests/       # run test suite
```

## 📁 Project Structure

```
├── kitaev.py              # All-in-one implementation
├── optimize.py            # Optimization interface
├── analysis.py            # Analysis: band dispersion & G(r)
├── plot/                  # 🎨 PRL-level scientific plotting
│   ├── correlation.py     # High-quality G(r) visualization
│   ├── dispersion.py      # High-quality band structure plots
│   └── __init__.py        # Plot module interface
├── docs/theory/           # Theory documentation
│   ├── majorana.md        # Majorana representation
│   ├── kernels.md         # Kitaev kernel theory
│   ├── correlator.md      # Gaussian mapping
│   └── permutation.md     # Implementation details
└── tests/test_core.py     # Test suite
```

## ⚙️ Parameters

### Physical Parameters
- `Jx, Jy, Jz`: Kitaev coupling strengths
- `Lx, Ly`: Momentum space sampling (k-point grid)
- `Nv`: Virtual bond dimension

### Technical Parameters  
- `max_iterations`: Optimization steps
- `seed`: Random initialization seed

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests include:
- Momentum space grid generation
- Majorana kernel properties  
- Gaussian correlator mapping
- Energy loss functions
- Full optimization pipeline

## 📚 Theory

The implementation is based on:
1. **Majorana representation** of Kitaev spins
2. **Gaussian fPEPS** ansatz for ground states
3. **Riemannian optimization** on Stiefel manifolds

See `docs/theory/` for detailed mathematical foundations.

## 🎨 Design Philosophy  

This codebase prioritizes:
- **Simplicity**: Single model, single file, zero ambiguity
- **Performance**: JAX JIT compilation, efficient algorithms  
- **Clarity**: Self-contained, well-documented code
- **Reproducibility**: Fixed algorithms, deterministic results

---

*Focused. Fast. Unambiguous.* 🎯
