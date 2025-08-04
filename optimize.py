#!/usr/bin/env python3
"""
Minimal command-line interface for Kitaev honeycomb spin-1/2 fPEPS.

Usage:
    python main.py                    # Use defaults
    python main.py --Jx 0.5 --Nv 3   # Override specific parameters
"""

import jax
import jax.numpy as jnp
import argparse
from kitaev import run_kitaev_simulation, DEFAULT_CONFIG
from initialize import run_local_initialization, create_global_system_from_local, save_and_get_info, run_auto_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kitaev honeycomb spin-1/2 fPEPS optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Initialization option
    parser.add_argument("--initialize", action='store_true',
                       help="Perform local Dirac point initialization before global optimization")
    parser.add_argument("--dirac_x", type=float, default=jnp.pi/3,
                       help="Dirac point x coordinate for initialization")
    parser.add_argument("--dirac_y", type=float, default=-jnp.pi/3,
                       help="Dirac point y coordinate for initialization")
    parser.add_argument("--init_radius", type=float, default=0.2,
                       help="Radius for local initialization")
    parser.add_argument("--local_Lx", type=int, default=21,
                       help="Local grid size in x for initialization")
    parser.add_argument("--local_Ly", type=int, default=21,
                       help="Local grid size in y for initialization")
    parser.add_argument("--init_iterations", type=int, default=2000,
                       help="Number of iterations for local initialization")
    
    # Auto-analysis option
    parser.add_argument("--auto_analysis", action='store_true',
                       help="Automatically run Dirac dense analysis after optimization")
    parser.add_argument("--analysis_radius", type=float, default=0.001,
                       help="Half-width of square region around Dirac point for dense analysis")
    parser.add_argument("--grid_size", type=int, default=32,
                       help="Number of points in each direction for dense analysis")
    
    # Add arguments from DEFAULT_CONFIG
    for key, default_value in DEFAULT_CONFIG.items():
        if isinstance(default_value, bool):
            # Handle boolean arguments properly
            parser.add_argument(
                f"--{key}", 
                action='store_true',
                default=default_value,
                help=f"{key} (default: {default_value})"
            )
            # Add --no-version for True defaults
            if default_value:
                parser.add_argument(
                    f"--no-{key}",
                    dest=key,
                    action='store_false', 
                    help=f"Disable {key}"
                )
        else:
            parser.add_argument(
                f"--{key}", 
                type=type(default_value), 
                default=default_value,
                help=f"{key}"
            )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    config = vars(args)
    
    print("Devices:", jax.devices())

    # Check if initialization is requested
    if args.initialize:
        print("🎯 STARTING WITH LOCAL DIRAC INITIALIZATION")
        
        # Prepare local configuration
        local_config = {
            "dirac_point": (args.dirac_x, args.dirac_y),
            "radius": args.init_radius,
            "local_Lx": args.local_Lx,
            "local_Ly": args.local_Ly,
            "Nv": args.Nv,
            "Jx": args.Jx,
            "Jy": args.Jy,
            "Jz": args.Jz,
            "max_iterations": args.init_iterations,
            "seed": args.seed,
            "verbosity": args.verbosity,
            "save_results": False,  # Don't save intermediate results
        }
        
        # Run local initialization
        local_system, local_result = run_local_initialization(local_config)
        
        print(f"🎯 LOCAL INITIALIZATION COMPLETE:")
        print(f"   Final local energy: {local_system.energy():.8f}")
        print(f"   Local iterations: {len(local_result.log['iterations']['cost'])}")
        print(f"\n🌍 CONTINUING WITH GLOBAL OPTIMIZATION...")
        
        # Create global system with locally optimized T matrix
        global_config = {k: v for k, v in config.items() 
                        if k not in ['initialize', 'dirac_x', 'dirac_y', 'init_radius', 
                                   'local_Lx', 'local_Ly', 'init_iterations',
                                   'auto_analysis', 'analysis_radius', 'grid_size']}
        
        global_system = create_global_system_from_local(local_system, global_config)
        
        # Replace the T matrix and run global optimization
        from kitaev import optimize
        kitaev_system, result = optimize(global_system, 
                                       max_iterations=args.max_iterations,
                                       verbosity=args.verbosity)
        
        print(f"🌍 GLOBAL OPTIMIZATION COMPLETE:")
        print(f"   Final global energy: {kitaev_system.energy():.8f}")
        print(f"   Global iterations: {len(result.log['iterations']['cost'])}")
        
        # Save final results if requested
        saved_info = None
        if args.save_results:
            saved_info = save_and_get_info(kitaev_system, result, global_config)
        
        # Run auto-analysis if requested
        if args.auto_analysis and saved_info:
            run_auto_analysis(
                saved_files_info=saved_info,
                dirac_point=(args.dirac_x, args.dirac_y),
                analysis_radius=args.analysis_radius,
                grid_size=args.grid_size
            )
        elif args.auto_analysis and not saved_info:
            print("⚠️ Auto-analysis skipped: results not saved (use --save_results)")
    
    else:
        print("🌍 STANDARD GLOBAL OPTIMIZATION")
        # Run standard simulation
        kitaev_system, result = run_kitaev_simulation(config)
        
        # Run auto-analysis if requested
        if args.auto_analysis and args.save_results:
            # For standard optimization, we need to find the saved files
            # The files are saved in results/NvXseedY/ with timestamp
            from pathlib import Path
            import glob
            import os
            
            results_dir = Path(config.get("workingdir", "./results"))
            seed_dir = results_dir / f"Nv{args.Nv}seed{args.seed}"
            
            if seed_dir.exists():
                # Find the most recent Glocal file
                glocal_files = list(seed_dir.glob("*_Glocal.npy"))
                if glocal_files:
                    # Use the most recent file
                    glocal_file = max(glocal_files, key=os.path.getctime)
                    
                    # Create a simple saved_info structure
                    saved_info = {
                        'seed_directory': str(seed_dir),
                        'files': {'Glocal_matrix': glocal_file.name}
                    }
                    
                    print(f"\n🔬 AUTO-ANALYSIS (Standard mode)")
                    run_auto_analysis(
                        saved_files_info=saved_info,
                        dirac_point=(jnp.pi/3, -jnp.pi/3),  # Use default Dirac point
                        analysis_radius=args.analysis_radius,
                        grid_size=args.grid_size
                    )
                else:
                    print("⚠️ Auto-analysis skipped: No Glocal files found")
            else:
                print("⚠️ Auto-analysis skipped: Results directory not found")
        elif args.auto_analysis and not args.save_results:
            print("⚠️ Auto-analysis skipped: results not saved (use --save_results)")
    
    # Print final result
    print(f"Final energy: {kitaev_system.energy():.8f}")
    return kitaev_system.energy()

if __name__ == '__main__':
    main()