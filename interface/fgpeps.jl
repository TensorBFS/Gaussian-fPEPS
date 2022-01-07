# Julia interface

# Basic
# parser.add_argument("--Nv", type=int, default=3, help="Nv")
# parser.add_argument("--Lx", type=int, default=101, help="system size length")
# parser.add_argument("--Ly", type=int, default=101, help="system size length")

# Hamiltonian
# parser.add_argument("--ht", type=float, default = 1.0, help = "Hoping term")
# parser.add_argument("--DeltaX", type=float, default = 1.0, help = "paring term in X direction")
# parser.add_argument("--DeltaY", type=float, default = -1.0, help = "paring term in Y direction")
# parser.add_argument("--Mu", type=float, default = 0.0, help = "Doping term")
# parser.add_argument("--delta",type=float, default = 0.1, help = "Doping hole")

# Optimizer is fixed to conj-grad,
# parser.add_argument("--MaxIter", type=int, default = 1000, help= "MaxIteration for ABD optimizer")
# parser.add_argument("--gtol", type=float, default=1E-7, help="g tol for optimizer")
# parser.add_argument("--OptimDisp", type=int, default=1, help="displya optimization information")

"""
resolve_fgpeps(;Nv::Int,Lx::Int,Ly::Int,ht::Number,DeltaX::Number,DeltaY::Number,Mu::Number,delta::Number,MaxIter::Int,gtol::Number)

Nv::Int, Number of complex fermions on each leg. Nv=1,2,3...
Lx::Int, square lattice Lx, suggest to be odd number
Ly::Int, square lattice Ly, suggest to be odd number
ht::Number, Hoping term
DeltaX::Number, Δx for BCS ansatz
DeltaY::Number, Δy for BCS ansatz
"""
function resolve_fgpeps(;Nv::Int,Lx::Int,Ly::Int,ht::Number,DeltaX::Number,DeltaY::Number,Mu::Number,delta::Number,MaxIter::Int,gtol::Number)
    print(Nv,Lx,Ly)

end


#------------------------------------
resolve_fgpeps(;Nv=2,Lx=2,Ly=2,ht=1.0,DeltaX=1.0,DeltaY=1.0,Mu=1.0,delta=1.0,MaxIter=100,gtol=1.0)