#!/usr/bin/env python3
"""
Isolated correlation analysis module for Kitaev fPEPS results.
Handles FFT-based real-space correlation measurements separately.
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

# Fixed parameters
NF = 1  # Number of physical fermion flavors (fixed for Kitaev)

# ============================================================================
# Core Functions (from analysis.py)
# ============================================================================

def batched_k(L):
    """Generate momentum points in [-π, π] range. Unified Lx=Ly=L."""
    X, Y = jnp.meshgrid(jnp.arange(L)/L, jnp.arange(L)/L)
    # Transform from [0,1] to [-π, π] range
    kx = 2 * jnp.pi * X  # range [0, 2π)
    ky = 2 * jnp.pi * Y  # range [0, 2π)
    return jnp.array([kx.flatten(), ky.flatten()]).T

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

def compute_correlator_from_glocal(Glocal, L, Nv):
    """
    Compute correlator from saved Glocal matrix. Unified L instead of Lx,Ly.
    
    Args:
        Glocal: Saved Glocal matrix (from optimization)
        L: k-space sampling parameters (unified)
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

def compute_real_space_correlations(correlator, k_points, L, max_distance=30):
    """
    Compute real-space correlation function G(r) via Fourier transform.
    L serves as both k-space sampling and real-space grid size (L=K).
    """
    K_grid = int(jnp.sqrt(len(k_points)))  # Current k-space grid size
    correlator_2d = correlator.reshape(K_grid, K_grid, 2*NF, 2*NF)
    
    # Zero-pad to L if needed
    if L > K_grid:
        pad = (L - K_grid) // 2
        pad_width = ((pad, L-K_grid-pad), (pad, L-K_grid-pad), (0,0), (0,0))
        correlator_padded = jnp.pad(correlator_2d, pad_width)
    else:
        correlator_padded = correlator_2d
        L = K_grid  # Use original size if L <= K_grid
    
    # Fourier transform to real space
    G_r_full = jnp.fft.ifft2(correlator_padded, axes=(0, 1))
    G_r_full = jnp.fft.fftshift(G_r_full, axes=(0, 1))
    
    # Extract central region around r=0
    center = L // 2
    x_slice = slice(max(0, center - max_distance), min(L, center + max_distance + 1))
    y_slice = slice(max(0, center - max_distance), min(L, center + max_distance + 1))
    G_r = G_r_full[x_slice, y_slice]
    
    # Generate real-space coordinate grid centered at (0,0)
    actual_max = min(max_distance, center, L - center - 1)
    r_coords = jnp.arange(-actual_max, actual_max + 1)
    r_points = jnp.stack(jnp.meshgrid(r_coords, r_coords, indexing='ij'), axis=-1)
    
    return G_r, r_points

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

def save_correlation_data(correlator, k_points, G_r, r_points, metadata, data_dir, K):
    """
    Save correlation data to ./data/ directory with descriptive filenames.
    K serves as both k-space sampling and real-space grid size.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metadata for filename
    config = metadata.get('config', {})
    Nv = config.get('Nv', 'unknown')
    seed = config.get('seed', 'unknown')
    Jx = config.get('Jx', 1.0)
    Jy = config.get('Jy', 1.0) 
    Jz = config.get('Jz', 1.0)
    
    # Create base filename with all parameters (L=K, so only show K)
    base_name = f"correlation_Nv{Nv}_seed{seed}_J{Jx:.1f}_{Jy:.1f}_{Jz:.1f}_K{K}"
    
    # Save correlation_01 component (Gamma matrices) with corresponding k_points
    correlator_01 = correlator[:, 0, 1]  # Extract (0,1) component  
    np.save(data_dir / f"{base_name}_correlator_01.npy", np.array(correlator_01))
    
    # Save k_points (CRITICAL for data maintenance - k coordinates for each G(k))
    np.save(data_dir / f"{base_name}_k_points.npy", np.array(k_points))
    
    # Save G(r) data
    np.save(data_dir / f"{base_name}_Gr.npy", np.array(G_r))
    np.save(data_dir / f"{base_name}_r_points.npy", np.array(r_points))
    
    # Save metadata
    correlation_metadata = {
        'original_config': config,
        'correlation_params': {
            'K': K,  # K serves as both k-space sampling and real-space grid size
            'correlator_shape': correlator.shape,
            'k_points_shape': k_points.shape,
            'Gr_shape': G_r.shape,
            'description': 'Correlation analysis data for Kitaev fPEPS',
            'data_files': {
                'correlator_01': f"{base_name}_correlator_01.npy",
                'k_points': f"{base_name}_k_points.npy", 
                'Gr': f"{base_name}_Gr.npy",
                'r_points': f"{base_name}_r_points.npy"
            }
        }
    }
    
    with open(data_dir / f"{base_name}_meta.json", 'w') as f:
        json.dump(correlation_metadata, f, indent=2)
    
    print(f"📁 Correlation data saved to: {data_dir}")
    print(f"   - {base_name}_correlator_01.npy: (K*K,) complex G(k) correlator matrices")
    print(f"   - {base_name}_k_points.npy: (K*K, 2) k-point coordinates for G(k)")
    print(f"   - {base_name}_Gr.npy: Real-space correlations G(r)")
    print(f"   - {base_name}_r_points.npy: Real-space coordinates")
    print(f"   - {base_name}_meta.json: Metadata with file mapping")

def plot_single_correlation_diagonal(G_r, r_points, max_distance, data_dir, base_name):
    """
    Generate only the diagonal correlation scaling plot for G_01 component.
    """
    from plot.correlation import plot_correlation_scaling_analysis_diagonal
    
    # Only plot G_01 component diagonal scaling
    save_path = data_dir / f"{base_name}_correlation_scaling_G01_diagonal.png"
    
    plot_correlation_scaling_analysis_diagonal(
        G_r=np.array(G_r), 
        r_points=np.array(r_points),
        component=(0,1),
        max_distance=max_distance,  # Show all points for visualization
        save_path=save_path
    )
    
    print(f"🎨 Generated single correlation plot: {save_path.name}")

# ============================================================================
# Main Correlation Analysis Pipeline
# ============================================================================

def run_correlation_analysis_single(glocal_file, K, max_distance=20, data_dir="./data", generate_plot=True):
    """
    Single K-value correlation analysis pipeline.
    
    Args:
        glocal_file: Path to Glocal file (.npy or .json)
        K: k-space sampling = real-space grid size (unified L=K)
        max_distance: maximum distance for G(r) calculation
        data_dir: data save directory
        generate_plot: whether to generate the single diagonal plot
    """
    print(f"🔬 Starting correlation analysis of: {glocal_file}")
    
    # Load Glocal and metadata
    Glocal, metadata = load_glocal_from_file(glocal_file)
    
    # Extract original parameters
    if metadata and 'config' in metadata:
        original_config = metadata['config']
        Nv = original_config['Nv']
        seed = original_config['seed']
        print(f"📋 Original parameters: Nv={Nv}, seed={seed}")
        print(f"📋 Correlation parameters: K={K} (k-space = real-space grid), max_distance={max_distance}")
    else:
        raise ValueError("Could not extract Nv from metadata. Please specify manually.")
    
    # Determine data directory path
    data_path = Path(data_dir)
    if glocal_file.startswith('results/'):
        # Extract relative path from results/ and put in data/
        relative_path = Path(glocal_file).parent.relative_to('results')
        data_path = data_path / relative_path
    
    # Compute correlator
    print("🔄 Computing correlator...")
    correlator, k_points = compute_correlator_from_glocal(Glocal, K, Nv)
    
    # Compute real-space correlations
    print("🌐 Computing real-space correlations...")
    G_r, r_points = compute_real_space_correlations(correlator, k_points, K, max_distance)
    
    # Save correlation data
    print("💾 Saving correlation data...")
    save_correlation_data(correlator, k_points, G_r, r_points, metadata, data_path, K)
    
    # Generate single diagonal plot if requested
    if generate_plot:
        config = metadata.get('config', {})
        Nv = config.get('Nv', 'unknown')
        seed = config.get('seed', 'unknown')
        Jx = config.get('Jx', 1.0)
        Jy = config.get('Jy', 1.0) 
        Jz = config.get('Jz', 1.0)
        base_name = f"correlation_Nv{Nv}_seed{seed}_J{Jx:.1f}_{Jy:.1f}_{Jz:.1f}_K{K}"
        
        print("🎨 Generating correlation diagonal plot...")
        plot_single_correlation_diagonal(G_r, r_points, max_distance, data_path, base_name)
    
    print(f"✅ Correlation analysis complete for K={K}!")
    return {
        'K': K,
        'correlator': correlator,
        'k_points': k_points,
        'G_r': G_r,
        'r_points': r_points,
        'data_path': data_path
    }

def run_correlation_analysis(glocal_file, K_list, max_distance=20, data_dir="./data", generate_plot=True, parallel=True):
    """
    Multi-K correlation analysis pipeline with parallel processing.
    
    Args:
        glocal_file: Path to Glocal file (.npy or .json)
        K_list: List of k-space sampling values [20, 30, 50, 100, ...] (L=K automatically)
        max_distance: maximum distance for G(r) calculation
        data_dir: data save directory
        generate_plot: whether to generate plots
        parallel: whether to use parallel processing
    
    Returns:
        dict: Results for all K values
    """
    if isinstance(K_list, int):
        # Single K value - convert to list
        K_list = [K_list]
    
    print(f"🔬 Starting correlation analysis for K values: {K_list}")
    print(f"📋 Parameters: L=K (unified), max_distance={max_distance}")
    print(f"⚙️  Parallel processing: {parallel}")
    
    results = {}
    
    if parallel and len(K_list) > 1:
        # Parallel processing
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            
            max_workers = min(len(K_list), mp.cpu_count())
            print(f"🚀 Using {max_workers} parallel workers")
            
            def run_single_k(K):
                return K, run_correlation_analysis_single(
                    glocal_file, K, max_distance, data_dir, generate_plot
                )
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(run_single_k, K): K for K in K_list}
                
                # Collect results as they complete
                for future in as_completed(futures):
                    K = futures[future]
                    try:
                        K_result, result = future.result()
                        results[f'K_{K}'] = result
                        print(f"✅ Completed K={K}")
                    except Exception as exc:
                        print(f"❌ K={K} generated an exception: {exc}")
                        results[f'K_{K}'] = {'error': str(exc)}
        
        except ImportError:
            print("⚠️  Parallel processing not available, falling back to sequential")
            parallel = False
    
    if not parallel:
        # Sequential processing
        print("🔄 Processing sequentially...")
        for K in K_list:
            print(f"\n🔄 Processing K={K}...")
            try:
                result = run_correlation_analysis_single(
                    glocal_file, K, max_distance, data_dir, generate_plot
                )
                results[f'K_{K}'] = result
                print(f"✅ Completed K={K}")
            except Exception as exc:
                print(f"❌ K={K} failed: {exc}")
                results[f'K_{K}'] = {'error': str(exc)}
    
    # Summary
    successful_K = [K for K in K_list if f'K_{K}' in results and 'error' not in results[f'K_{K}']]
    failed_K = [K for K in K_list if f'K_{K}' not in results or 'error' in results[f'K_{K}']]
    
    print(f"\n📈 Multi-K Correlation Analysis Summary:")
    print(f"   Successful K values: {successful_K}")
    if failed_K:
        print(f"   Failed K values: {failed_K}")
    print(f"   Total data sets: {len(successful_K)}")
    
    # Save summary metadata
    if successful_K:
        summary_path = Path(data_dir)
        if glocal_file.startswith('results/'):
            relative_path = Path(glocal_file).parent.relative_to('results')
            summary_path = summary_path / relative_path
        
        summary = {
            'glocal_file': glocal_file,
            'K_values': K_list,
            'successful_K': successful_K,
            'failed_K': failed_K,
            'max_distance': max_distance,
            'analysis_type': 'multi_K_correlation',
            'note': 'L=K (k-space sampling = real-space grid size)'
        }
        
        with open(summary_path / "multi_K_correlation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📁 Summary saved: {summary_path}/multi_K_correlation_summary.json")
    
    return results

# ============================================================================
# Command Line Interface  
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Isolated correlation analysis for Kitaev fPEPS results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("glocal_file", type=str, 
                       help="Path to Glocal .npy file or metadata .json file")
    parser.add_argument("--K", type=str, default="50",
                       help="k-space sampling (also sets L=K for real-space grid). Single value (50) or list (20,30,50,100)")
    parser.add_argument("--max_distance", type=int, default=20,
                       help="Maximum distance for G(r) calculation")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Data save directory")
    parser.add_argument("--no_plot", action="store_true",
                       help="Skip generating correlation plot")
    parser.add_argument("--no_parallel", action="store_true",
                       help="Disable parallel processing (use sequential)")
    return parser.parse_args()

def parse_K_values(K_str):
    """Parse K values from string input."""
    try:
        if ',' in K_str:
            # List of values: "20,30,50,100"
            K_list = [int(k.strip()) for k in K_str.split(',')]
        else:
            # Single value: "50"
            K_list = [int(K_str)]
        return K_list
    except ValueError:
        raise ValueError(f"Invalid K value format: {K_str}. Use single value (50) or comma-separated list (20,30,50)")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse K values
    try:
        K_list = parse_K_values(args.K)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"🔬 K values to process: {K_list}")
    
    results = run_correlation_analysis(
        glocal_file=args.glocal_file,
        K_list=K_list,
        max_distance=args.max_distance,
        data_dir=args.data_dir,
        generate_plot=not args.no_plot,
        parallel=not args.no_parallel
    )
    
    # Enhanced summary
    successful_K = [int(k.split('_')[1]) for k in results.keys() if 'error' not in results[k]]
    
    print(f"\n📈 Final Correlation Analysis Summary:")
    print(f"   Processed K values: {K_list}")
    print(f"   Successful: {successful_K}")
    print(f"   Grid size: K×K (k-space = real-space)")
    print(f"   Max distance: {args.max_distance}")
    print(f"   Parallel processing: {not args.no_parallel}")
    
    if successful_K:
        print(f"   📁 Data files generated for each K value in: {args.data_dir}")
        print(f"   📝 Summary file: multi_K_correlation_summary.json")

if __name__ == '__main__':
    main() 