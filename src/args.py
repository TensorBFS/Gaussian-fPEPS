import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("--folder", default='./data/',help="where to store results")
parser.add_argument("--Nv", type=int, default=3, help="Nv")
parser.add_argument("--Lx", type=int, default=11, help="system size length")
parser.add_argument("--Ly", type=int, default=11, help="system size length")

# Hamiltonian
parser.add_argument("--ht", type=float, default = 1.0, help = "Hoping term")
parser.add_argument("--DeltaX", type=float, default = 1.0, help = "paring term in X direction")
parser.add_argument("--DeltaY", type=float, default = -1.0, help = "paring term in Y direction")

# Optimizer 
parser.add_argument("--optimizer", default='trust-ncg', choices=['trust-ncg','None'], help="optimizer")
parser.add_argument("--MaxIter", type=int, default = 1000, help= "MaxIteration for ABD optimizer")
parser.add_argument("--gtol", type=float, default=1E-7, help="g tol for optimizer")
parser.add_argument("--OptimDisp", type=int, default=1, help="displya optimization information")

# load and write file
# parser.add_argument("-load", default="temp.h5", help="load")
# parser.add_argument("-write", default="temp.h5", help="write")
parser.add_argument("--label", type=int, default=999,help= "label for multi running")
parser.add_argument("--loadlabel", type=int, default=999,help= "label for multi running")

args = parser.parse_args()