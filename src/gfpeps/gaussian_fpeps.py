import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit
from .loss import optimize_runtime_loss, energy_function
from .loadwrite import initialT,savelog,savelog_trivial
from .exact import eg
from .deltatomu import solve_mu
from .measure import measure
import logging

# import Manopt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions,ConjugateGradient

def gaussian_fpeps(cfg):
    # unpack cfg
    np.random.seed(cfg.params.seed)
    Nv = cfg.params.Nv
    Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
    LoadKey, WriteKey  = cfg.file.LoadFile, cfg.file.WriteFile
    
    cfgh = cfg.hamiltonian
    ht = cfgh.ht
    DeltaX, DeltaY = cfgh.DeltaX, cfgh.DeltaY
    delta, Mu = cfgh.delta, cfgh.Mu
    pairing_type = getattr(cfgh, 'pairing_type', 'd_wave')  # Default to d_wave if not specified
    
    # Print configuration summary
    print("=" * 80)
    print("Gaussian fPEPS Configuration Summary")
    print("=" * 80)
    print(f"System Parameters:")
    print(f"  - Virtual fermions (Nv): {Nv}")
    print(f"  - System size: {Lx} x {Ly}")
    print(f"  - Random seed: {cfg.params.seed}")
    print()
    print(f"Hamiltonian Parameters:")
    print(f"  - Hopping amplitude (ht): {ht}")
    print(f"  - Pairing type: {pairing_type}")
    print(f"  - DeltaX: {DeltaX}")
    print(f"  - DeltaY: {DeltaY}")
    print(f"  - Chemical potential (Mu): {Mu}")
    print(f"  - Hole density (delta): {delta}")
    print(f"  - Solve Mu from delta: {cfgh.solve_mu_from_delta}")
    print()
    print(f"File Configuration:")
    print(f"  - Load file: {LoadKey}")
    print(f"  - Write file: {WriteKey}")
    print(f"  - Save each step: {cfg.file.SaveEachSteps}")
    print()
    print(f"Optimization Parameters:")
    print(f"  - Max iterations: {cfg.optimizer.MaxIter}")
    print(f"  - Gradient tolerance: {cfg.optimizer.gtol}")
    print(f"  - Backend: {cfg.backend}")
    print("=" * 80)
    print()

    Tsize = 8*Nv+4
    T = initialT(LoadKey,Tsize)
    U,S,V = np.linalg.svd(T)
    T = U @ V

    if cfgh.solve_mu_from_delta:
        logging.info("Overwrite Origin Mu")
        Mu = solve_mu(DeltaX,delta)
    
    lossT = jit(optimize_runtime_loss(Nv=Nv,Lx=Lx,Ly=Ly,
    hoping=ht,DeltaX=DeltaX,DeltaY=DeltaY,Mu=Mu,pairing_type=pairing_type), backend=cfg.backend)
    
    def egrad(x): return np.array(jax.grad(lossT)(jnp.array(x)))

    @jax.jit
    def hvp_loss(primals, tangents): return jax.jvp(jax.grad(lossT), primals, tangents)[1]

    def ehessa(x,v): return np.array(hvp_loss((jnp.array(x),), (jnp.array(v),)))

    # Calculate exact ground state energy
    Eg = eg(Lx, Ly, ht, DeltaX, DeltaY, Mu, pairing_type=pairing_type) # Will use solved Mu to calculate Eg
    logging.info("Exact ground state energy Eg = {}\n".format(Eg))
    
    # Print pairing type specific information
    if pairing_type == 'p_ip_wave':
        print("P+IP Wave Pairing Details:")
        print(f"  - Real part: Δ_X * sin(kx) - Δ_Y * sin(ky) = {DeltaX} * sin(kx) - {DeltaY} * sin(ky)")
        print(f"  - Imaginary part: Δ_X * sin(ky) + Δ_Y * sin(kx) = {DeltaX} * sin(ky) + {DeltaY} * sin(kx)")
        print(f"  - Complex pairing: Δ(k) = ({DeltaX}*sin(kx) - {DeltaY}*sin(ky)) + i*({DeltaX}*sin(ky) + {DeltaY}*sin(kx))")
        print("  - Physical significance: Chiral pairing breaks time-reversal symmetry")
        print("  - Topological properties: Supports Majorana fermions at edges/defects")
        print()
    elif pairing_type == 'd_wave':
        print("D-Wave Pairing Details:")
        print(f"  - Pairing function: Δ(k) = {DeltaX} * cos(kx) - {DeltaY} * cos(ky)")
        print("  - Physical significance: Nodal structure, conventional superconductor")
        print()
    elif pairing_type == 's_wave':
        print("S-Wave Pairing Details:")
        print(f"  - Pairing function: Δ(k) = {DeltaX} (constant)")
        print("  - Physical significance: Full gap, isotropic pairing")
        print()
    # Optimizer

    manifold = Stiefel(Tsize, Tsize)
    @pymanopt.function.numpy(manifold)
    def cost(x):
        return lossT(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(x):
        return egrad(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(x,y):
        return ehessa(x,y)


    problem = Problem(manifold=manifold, 
                      cost=cost,
                      euclidean_gradient=euclidean_gradient,
                      euclidean_hessian=euclidean_hessian)
    solver = ConjugateGradient(log_verbosity=1, max_iterations=cfg.optimizer.MaxIter)

    print("Starting optimization...")
    print(f"Initial tensor size: {Tsize} x {Tsize}")
    print(f"Optimization method: Conjugate Gradient")
    print(f"Maximum iterations: {cfg.optimizer.MaxIter}")
    print(f"Gradient tolerance: {cfg.optimizer.gtol}")
    print()
    
    result = solver.run(problem, initial_point=T)
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])
    
    print("Optimization Progress:")
    print("Iteration    Cost                    Gradient Norm")
    print("---------    -----------------------  -------------")
    for iter in range(len(log_cost)):
        print(f"{iter:9d}    {log_cost[iter]:20.8e}    {log_gnorm[iter]:12.6e}")

    print()
    print("Optimization Results:")
    print(f"  - Final cost: {result.cost:.8e}")
    print(f"  - Final gradient norm: {result.gradient_norm:.8e}")
    print(f"  - Total iterations: {len(log_cost)}")
    print(f"  - Convergence: {'Yes' if result.gradient_norm < cfg.optimizer.gtol else 'No'}")
    print()
    
    # measure final result
    # rhoup, rhodn, kappa = measure(cfg,result.point)
    
    Xopt = result.point
    final_cost = lossT(Xopt)
    args = {"Mu":Mu,"DeltaX":DeltaX,"DeltaY":DeltaY,"delta":delta,
            "ht":ht,"Lx":Lx,"Ly":Ly,"Nv":Nv,"seed":cfg.params.seed}
    
    # Measure final observables
    rhoup, rhodn, kappa = measure(cfg, Xopt)
    
    print("Final Results:")
    print(f"  - Final energy: {final_cost:.8e}")
    print(f"  - Exact ground state energy: {Eg:.8e}")
    print(f"  - Energy difference: {abs(final_cost - Eg):.8e}")
    print(f"  - Average particle density: {np.mean(rhoup + rhodn):.6f}")
    print(f"  - Average pairing amplitude: {np.mean(np.abs(kappa)):.6f}")
    print()
    
    print("Saving results...")
    savelog_trivial(WriteKey, Xopt, final_cost, Eg, args, (rhoup, rhodn, kappa))
    print(f"  - Results saved to: {WriteKey}")
    
    if cfg.file.SaveEachSteps:
        print("  - Saving intermediate steps...")
        for iter in range(len(log_cost)):
            Xopt = np.array(result.log["iterations"]["point"])[iter]
            savelog_trivial(WriteKey[:-3]+f"-iter{iter}"+WriteKey[-3:], Xopt, lossT(Xopt), Eg, args, measure(cfg, Xopt))
        print(f"  - Intermediate steps saved")
    
    return Xopt