from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax.scipy.linalg import expm

def _j_matrix(Nv, Nf=2):
    r"""Create the symplectic form matrix J.
    
    Creates a block diagonal matrix of 2x2 antisymmetric blocks [0, 1; -1, 0].
    The size is determined by the number of virtual modes (8*Nv) plus physical fermions (2*Nf).
    Virtual modes are split into x and y directions (2*Nv each).
    
    Matrix structure:
    - Total size: (2*Nf + 8*Nv) × (2*Nf + 8*Nv)
    - First 2*Nf rows/cols: Physical fermions
    - Next 2*Nv rows/cols: Virtual modes in x direction
    - Last 2*Nv rows/cols: Virtual modes in y direction
    
    Args:
        Nv (int): Number of virtual modes per direction (x or y).
        Nf (int, optional): Number of physical fermions. Defaults to 2 (↑,↓).
    
    Returns:
        jnp.ndarray: Block diagonal matrix of shape ((2*Nf + 8*Nv), (2*Nf + 8*Nv))
            containing antisymmetric 2x2 blocks [0, 1; -1, 0].
    """
    return block_diag(*[jnp.array([[0,1.0],[-1,0]]) for i in range(Nf + 4*Nv)])

def _gamma_projector(T, J, Nv, Nf=2):
    r"""Compute the Gamma projector from orthogonal matrix T.
    
    The Gamma projector represents the local fermionic state in terms of physical
    and virtual modes. It is constructed as T^T J T where T is an orthogonal matrix
    that mixes physical and virtual modes.
    
    Matrix dimensions:
    - T: (2*Nf + 8*Nv) × (2*Nf + 8*Nv)
        * First 2*Nf rows/cols: Physical fermions
        * Next 2*Nv rows/cols: Virtual modes in x direction
        * Last 2*Nv rows/cols: Virtual modes in y direction
    - J: Same size as T, block diagonal with [0,1;-1,0] blocks
    
    Args:
        T (jnp.ndarray): Orthogonal matrix mixing physical and virtual modes.
        J (jnp.ndarray): Symplectic form matrix.
        Nv (int): Number of virtual modes per direction.
        Nf (int, optional): Number of physical fermions. Defaults to 2.
    
    Returns:
        jnp.ndarray: Gamma projector matrix T^T J T with same shape as T.
    """
    return jnp.transpose(T) @ J @ T

def getGammaProjector(T, Nv, Nf=2):
    r"""Create the local Gamma projector from an orthogonal matrix T.
    
    This function constructs the Gamma projector that represents the local fermionic
    state in terms of physical and virtual modes. The projector preserves the
    canonical anticommutation relations of the fermionic operators.
    
    Structure:
    1. Physical space: 2*Nf modes (e.g., for spin-1/2: c↑, c↓)
    2. Virtual space: 8*Nv modes
       - 2*Nv modes for x direction bonds
       - 2*Nv modes for y direction bonds
    
    Matrix dimensions:
    - Input T: (2*Nf + 8*Nv) × (2*Nf + 8*Nv)
    - Output Γ: Same size as T
    
    The resulting projector has a block structure:
    [A  B]
    [B† D]
    where:
    - A (2*Nf × 2*Nf): Physical-physical correlations
    - B (2*Nf × 8*Nv): Physical-virtual correlations
    - D (8*Nv × 8*Nv): Virtual-virtual correlations
    
    Args:
        T (jnp.ndarray): Orthogonal matrix of shape ((2*Nf + 8*Nv), (2*Nf + 8*Nv))
        Nv (int): Number of virtual modes per direction (x or y)
        Nf (int, optional): Number of physical fermions. Defaults to 2.
    
    Returns:
        jnp.ndarray: Gamma projector matrix of same shape as T
    """
    return _gamma_projector(T, _j_matrix(Nv, Nf), Nv, Nf)