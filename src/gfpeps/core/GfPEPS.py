import jax
from dataclasses import dataclass

@dataclass
class GaussianfPEPS(object):
    """A Runtime Object to define everything you need to run Gaussian fPEPS calculation, store everything you need.
    
    Attributes:
        Lx (int): Number of k-points in x direction for momentum space sampling
        Ly (int): Number of k-points in y direction for momentum space sampling
        Nf (int): Number of physical fermion flavors per site, set by kernels
        Nv (int): Number of virtual modes per bond

        kernel_names (str): Name of the kernel function
        kernel_params (dict): Additional parameters for the kernel, override the default ones

    """
    # Lattice parameters
    Lx: int = None
    Ly: int = None
    Nf: int = None
    Nv: int = None

    kernel_names: str = None 
    kernel_params = None

    kernel_functions: callable = None

    def __init__(self, Lx, Ly, kernel_params=None, *, Nv=None, kernel_names=None, seed=123):
        self.Lx = Lx
        self.Ly = Ly
        self.Nv = Nv
        self.kernel_names = kernel_names
        self.kernel_params = kernel_params
        kernel_info = set_kernel(kernel_names)
        self.kernel_functions = kernel_info[0]
        self.Nf = kernel_info[1]

        self.T = initialize_random_T(seed, 2*self.Nf + 8*self.Nv)

        self.loss = make_loss(self.kernel_functions, Lx=Lx, Ly=Ly, Nf=self.Nf, Nv=self.Nv, **self.kernel_params)

def initialize_random_T(seed, dim):
    T = jnp.array(jax.random.normal(key=jax.random.PRNGKey(seed), shape=(dim, dim)))
    U, S, V = jnp.linalg.svd(T)
    T = U @ V
    return T