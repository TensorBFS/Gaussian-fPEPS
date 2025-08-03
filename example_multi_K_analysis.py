#!/usr/bin/env python3
"""
Example script for multi-K correlation analysis.
Demonstrates how to analyze different k-space resolutions in parallel.
"""

import numpy as np
from analysis_correlation import run_correlation_analysis

def example_usage():
    """Example of how to use the multi-K correlation analysis."""
    
    # Example Glocal file
    glocal_file = "results/Nv2seed4/kitaev_Jx1.0_Jy1.0_Jz1.0_20250802_144255_Glocal.npy"
    
    # Define K values to analyze (convergence study)
    K_values = [20, 30, 50, 80, 100, 150, 200]
    
    print(f"🔬 Running multi-K correlation analysis")
    print(f"📊 K values: {K_values}")
    
    # Run parallel analysis
    results = run_correlation_analysis(
        glocal_file=glocal_file,
        K_list=K_values,
        max_distance=30,
        data_dir="./data",
        generate_plot=True,  # Generate plots for each K
        parallel=True  # Use parallel processing
    )
    
    # Analyze convergence
    print("\n📈 Convergence Analysis:")
    for K in K_values:
        key = f'K_{K}'
        if key in results and 'error' not in results[key]:
            correlator = results[key]['correlator']
            print(f"   K={K}: correlator shape = {correlator.shape}")
        else:
            print(f"   K={K}: FAILED")
    
    return results

def generate_K_series(start=20, end=800, step_type='geometric'):
    """Generate a series of K values for convergence study."""
    if step_type == 'geometric':
        # Geometric progression: 20, 30, 45, 67, 100, 150, 225, 337, 506, 759
        K_values = []
        K = start
        while K <= end:
            K_values.append(int(K))
            K *= 1.5  # Geometric factor
    elif step_type == 'linear':
        # Linear progression: 20, 40, 60, 80, ..., 800
        K_values = list(range(start, end + 1, (end - start) // 20))
    elif step_type == 'powers_of_2':
        # Powers of 2: 16, 32, 64, 128, 256, 512
        K_values = [2**i for i in range(4, 10) if 2**i <= end]
    else:
        # Custom list
        K_values = [20, 30, 50, 80, 100, 150, 200, 300, 400, 600, 800]
    
    return K_values

if __name__ == '__main__':
    # Different K series options
    print("🔬 K series options:")
    print(f"   Geometric: {generate_K_series(20, 800, 'geometric')}")
    print(f"   Linear: {generate_K_series(20, 800, 'linear')[:10]}...")  # Show first 10
    print(f"   Powers of 2: {generate_K_series(20, 800, 'powers_of_2')}")
    
    # You can run example_usage() here if you have a valid Glocal file
    print("\n💡 To run analysis:")
    print("   python analysis_correlation.py your_file.npy --K 20,30,50,100,200  # Multi-K (L=K)")
    print("   python analysis_correlation.py your_file.npy --K 200  # Single high-res K")
    print("   python analysis_correlation.py your_file.npy --K 20,30,50 --no_parallel  # Sequential")
    print("   python analysis_correlation.py your_file.npy --K 50,100,200,400 --max_distance 50  # Long-range") 