#!/usr/bin/env python3
import argparse
from gfpeps.run import run

def parse_args():
    parser = argparse.ArgumentParser(description='Gaussian fPEPS ground state optimization')
    
    # Model selection
    parser.add_argument('--model', type=str, default='kitaev_honeycomb_1half',
                      choices=['kitaev_honeycomb_1half', 'bcs_pairing_square'],
                      help='Physical model to simulate')
    
    # Lattice parameters
    parser.add_argument('--Lx', type=int, default=50,
                      help='Number of k-points in x direction')
    parser.add_argument('--Ly', type=int, default=50,
                      help='Number of k-points in y direction')
    parser.add_argument('--Nv', type=int, default=2,
                      help='Number of virtual modes')
    
    # Optimization parameters
    parser.add_argument('--max-iter', type=int, default=1000,
                      help='Maximum number of optimization iterations')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for initialization')
    
    # Model specific parameters
    parser.add_argument('--Jx', type=float, default=1.0,
                      help='Coupling Jx for Kitaev model')
    parser.add_argument('--Jy', type=float, default=1.0,
                      help='Coupling Jy for Kitaev model')
    parser.add_argument('--Jz', type=float, default=1.0,
                      help='Coupling Jz for Kitaev model')
    parser.add_argument('--t', type=float, default=1.0,
                      help='Hopping parameter for BCS model')
    parser.add_argument('--delta', type=float, default=0.5,
                      help='Pairing strength for BCS model')
    parser.add_argument('--mu', type=float, default=0.0,
                      help='Chemical potential for BCS model')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Prepare kernel parameters based on model
    if args.model == 'kitaev_honeycomb_1half':
        kernel_params = {'Jx': args.Jx, 'Jy': args.Jy, 'Jz': args.Jz}
    else:  # bcs_pairing_square
        kernel_params = {'t': args.t, 'delta': args.delta, 'mu': args.mu}
    
    # Run simulation
    run(
        Lx=args.Lx,
        Ly=args.Ly,
        Nv=args.Nv,
        kernel_names=args.model,
        kernel_params=kernel_params,
        max_iterations=args.max_iter,
        seed=args.seed
    )
    
    return 0

if __name__ == '__main__':
    main()