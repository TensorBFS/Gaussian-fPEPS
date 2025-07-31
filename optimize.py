#!/usr/bin/env python3
"""
Minimal command-line interface for Kitaev honeycomb spin-1/2 fPEPS.

Usage:
    python main.py                    # Use defaults
    python main.py --Jx 0.5 --Nv 3   # Override specific parameters
"""

import argparse
from kitaev import run_kitaev_simulation, DEFAULT_CONFIG

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kitaev honeycomb spin-1/2 fPEPS optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments from DEFAULT_CONFIG
    for key, default_value in DEFAULT_CONFIG.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{key}", 
                action='store_true' if not default_value else 'store_false',
                help=f"{key}"
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
    
    # Run simulation
    kitaev_system, result = run_kitaev_simulation(config)
    
    # Print final result
    print(f"Final energy: {kitaev_system.energy():.8f}")
    return kitaev_system.energy()

if __name__ == '__main__':
    main()