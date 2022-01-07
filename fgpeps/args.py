import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("--rfile", default='./data/noname.h5',help="where to store results. h5file")
parser.add_argument("--wfile", default='./data/noname.h5',help="where to store results. h5file")
parser.add_argument("--Nv", type=int, default=3, help="Nv")
parser.add_argument("--Lx", type=int, default=101, help="system size length")
parser.add_argument("--Ly", type=int, default=101, help="system size length")

# Hamiltonian
parser.add_argument("--ht", type=float, default = 1.0, help = "Hoping term")
parser.add_argument("--DeltaX", type=float, default = 1.0, help = "paring term in X direction")
parser.add_argument("--DeltaY", type=float, default = -1.0, help = "paring term in Y direction")
parser.add_argument("--Mu", type=float, default = 0.0, help = "Doping term")
parser.add_argument("--delta",type=float, default = 0.1, help = "Doping hole")

# Optimizer 
parser.add_argument("--optimizer", default='conj-grad', choices=['trust-ncg','conj-grad'], help="optimizer")
parser.add_argument("--MaxIter", type=int, default = 1000, help= "MaxIteration for ABD optimizer")
parser.add_argument("--gtol", type=float, default=1E-7, help="g tol for optimizer")
parser.add_argument("--OptimDisp", type=int, default=1, help="displya optimization information")

# load and write file)
parser.add_argument("--seed", type=int, default=999,help= "Seeds")

args = parser.parse_args()