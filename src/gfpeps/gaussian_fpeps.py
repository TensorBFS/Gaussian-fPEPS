import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from .ABD import getGammaProjector, getABD
from .Gin import BatchGammaIn, BatchK
from .GaussianLinearMap import GaussianLinearMap
from .measure import compute_correlators
from .loss import hubbard_bcs_kernel, energy_expectation

class GaussianFPEPS:
    r"""Gaussian Fermionic Projected Entangled Pair States (gfPEPS).
    
    This class implements a variational ansatz for fermionic many-body systems using
    Gaussian states. The ansatz consists of:
    1. Local fermionic modes (physical + virtual)
    2. Projectors that mix physical and virtual modes
    3. Virtual mode contractions between neighboring sites
    
    Structure per site:
    - Physical modes: 2*Nf (e.g., for spin-1/2: c↑, c↓)
    - Virtual modes: 8*Nv total
      * 2*Nv modes for x-direction bonds
      * 2*Nv modes for y-direction bonds
    
    Matrix dimensions:
    - T matrix: (2*Nf + 8*Nv) × (2*Nf + 8*Nv)
    - Gamma projector: Same size as T
    - A block: (2*Nf) × (2*Nf)
    - B block: (2*Nf) × (8*Nv)
    - D block: (8*Nv) × (8*Nv)
    
    Args:
        Lx (int): Number of sites in x direction
        Ly (int): Number of sites in y direction
        Nv (int): Number of virtual modes per direction (x or y)
        Nf (int, optional): Number of physical fermions per site. Defaults to 2.
    """
    
    def __init__(self, Lx, Ly, Nv, Nf=2):
        self.Lx = Lx
        self.Ly = Ly
        self.Nv = Nv
        self.Nf = Nf
        self.n = 2*Nf + 8*Nv  # Total modes per site
        
        # Initialize batch objects for momentum space calculations
        self.batch_k = BatchK(Lx, Ly)
        self.batch_gin = BatchGammaIn(Lx, Ly, Nv)
    
    def random_init_T(self, key):
        """Initialize random orthogonal T matrix.
        
        Args:
            key (jax.random.PRNGKey): JAX random key
            
        Returns:
            jnp.ndarray: Random orthogonal matrix of shape (n, n)
                where n = 2*Nf + 8*Nv
        """
        from jax.random import normal
        T = normal(key, (self.n, self.n))
        Q, R = jnp.linalg.qr(T)
        return Q
    
    def get_gamma_projector(self, T):
        """Compute Gamma projector from T matrix.
        
        Args:
            T (jnp.ndarray): Orthogonal matrix of shape (n, n)
            
        Returns:
            jnp.ndarray: Gamma projector of shape (n, n)
        """
        return getGammaProjector(T, self.Nv, self.Nf)
    
    def get_ABD(self, GammaP):
        """Extract A, B, D blocks from Gamma projector.
        
        Args:
            GammaP (jnp.ndarray): Gamma projector matrix
            
        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
                A ((2*Nf)×(2*Nf)), B ((2*Nf)×(8*Nv)), D ((8*Nv)×(8*Nv))
        """
        return getABD(GammaP, self.Nf)
    
    def compute_correlators(self, T):
        """Compute momentum space correlators G(k) and F(k).
        
        Process:
        1. Get Gamma projector from T
        2. Apply Gaussian linear map to get BatchGout
        3. Compute correlators from BatchGout
        
        Args:
            T (jnp.ndarray): Orthogonal matrix of shape (n, n)
            
        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: G(k) and F(k) correlators
                G(k): Normal correlators ⟨c†(k)c(k)⟩
                F(k): Anomalous correlators ⟨c(k)c(-k)⟩
        """
        GammaP = self.get_gamma_projector(T)
        BatchGout = GaussianLinearMap(GammaP, self.batch_gin)
        return compute_correlators(BatchGout)
    
    def energy_expectation(self, T, t, Dx, Dy, mu):
        """Compute energy expectation value for Hubbard-BCS Hamiltonian.
        
        H = Σ_k [c†(k) E(k) c(k) + c(k) D(k) c(-k) + h.c.]
        where:
        - E(k): Normal hopping terms
        - D(k): Pairing terms
        
        Args:
            T (jnp.ndarray): Orthogonal matrix of shape (n, n)
            t (float): Hopping strength
            Dx (float): x-direction pairing
            Dy (float): y-direction pairing
            mu (float): Chemical potential
            
        Returns:
            float: Energy expectation value ⟨H⟩
        """
        # Get momentum-dependent kernels
        E_k, D_k = vmap(lambda k: hubbard_bcs_kernel(k, t, Dx, Dy, mu))(self.batch_k)
        
        # Get correlators
        G, F = self.compute_correlators(T)
        
        # Compute energy
        return energy_expectation(G, F, E_k, D_k)

def gaussian_fpeps(cfg):
    # unpack cfg
    np.random.seed(cfg.params.seed)
    Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
    Nv = cfg.params.Nv
    Nf = cfg.params.Nf
    LoadKey, WriteKey  = cfg.file.LoadFile, cfg.file.WriteFile
    
    cfgh = cfg.hamiltonian
    ht = cfgh.ht
    DeltaX, DeltaY = cfgh.DeltaX, cfgh.DeltaY
    delta, Mu = cfgh.delta, cfgh.Mu

    gfpeps = GaussianFPEPS(Lx, Ly, Nv)
    Tsize = 2*Nf + 8*Nv
    T = gfpeps.random_init_T(np.random.PRNGKey(cfg.params.seed))
    U,S,V = np.linalg.svd(T)
    T = U @ V

    if cfgh.solve_mu_from_delta:
        logging.info("Overwrite Origin Mu")
        Mu = solve_mu(DeltaX,delta)
    
    lossT = jit(lambda T: gfpeps.energy_expectation(T, ht, DeltaX, DeltaY, Mu), backend=cfg.backend)
    
    def egrad(x): return np.array(jax.grad(lossT)(jnp.array(x)))

    @jax.jit
    def hvp_loss(primals, tangents): return jax.jvp(jax.grad(lossT), primals, tangents)[1]

    def ehessa(x,v): return np.array(hvp_loss((jnp.array(x),), (jnp.array(v),)))

    Eg = gfpeps.energy_expectation(T, ht, DeltaX, DeltaY, Mu) # Will use solved Mu to calculate Eg
    logging.info("Eg = {}\n".format(Eg))
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

    result = solver.run(problem, initial_point=T)
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])
    
    logging.info("Iterations \t Cost \t Gradient Norm")
    for iter in range(len(log_cost)):
        logging.info(f"{iter} \t {log_cost[iter]} \t {log_gnorm[iter]}")

    logging.info(f"Optimization done!, final cost: {result.cost}, gnorm: {result.gradient_norm }")
    
    # measure final result
    # rhoup, rhodn, kappa = measure(cfg,result.point)
    
    Xopt = result.point
    args = {"Mu":Mu,"DeltaX":DeltaX,"DeltaY":DeltaY,"delta":delta,
            "ht":ht,"Lx":Lx,"Ly":Ly,"Nv":Nv,"seed":cfg.params.seed}
    # savelog_trivial(WriteKey,Xopt,lossT(Xopt),Eg,args, measure(cfg,Xopt))
    
    if cfg.file.SaveEachSteps:
        for iter in range(len(log_cost)):
            Xopt = np.array(result.log["iterations"]["point"])[iter]
            # savelog_trivial(WriteKey[:-3]+f"-iter{iter}"+WriteKey[-3:],Xopt,lossT(Xopt), Eg, args, measure(cfg, Xopt))
    return Xopt