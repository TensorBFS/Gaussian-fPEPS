from jax import vmap
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

def batched_k(Lx, Ly):
    """Generate momentum points in the Brillouin zone for APBC-PBC boundary conditions.
    
    Args:
        Lx (int): Number of points in x direction.
        Ly (int): Number of points in y direction.
    
    Returns:
        jnp.ndarray: Array of shape (Lx*Ly, 2) containing momentum points.
    """
    X, Y = jnp.meshgrid((jnp.arange(Lx)-0.5)/Lx, jnp.arange(Ly)/Ly)
    return 2 * jnp.pi * jnp.array([X.flatten(), Y.flatten()]).T

def batched_Gin(Lx, Ly, Nv):
    """Generate batch of Gamma_in matrices for all momentum points in Brillouin zone.
    
    Args:
        Lx (int): Number of points in x direction.
        Ly (int): Number of points in y direction.
        Nv (int): Number of virtual modes per site.
    
    Returns:
        jnp.ndarray: Array of Gamma_in matrices, shape (Lx*Ly, 4*Nv, 4*Nv).
    """
    def _gamma_in(k, Nv):
        """Create Gamma_in matrix for given momentum k and Nv virtual modes.
        
        Args:
            k (array-like): Array of momentum values in the Brillouin zone.
            Nv (int): Number of virtual modes per site.
        
        Returns:
            jnp.ndarray: Block diagonal matrix of Gamma_in matrices.
        """
        def single_gamma(ki):
            t = jnp.exp(1j*ki)
            ct = -jnp.exp(-1j*ki)
            base = jnp.array([[0,0,0,t],[0,0,t,0],[0,ct,0,0],[ct,0,0,0]])
            return block_diag(*[base for _ in range(Nv)])
        
        return block_diag(*[single_gamma(ki) for ki in k])

    return vmap(lambda k: _gamma_in(k, Nv), 0)(batched_k(Lx, Ly))
