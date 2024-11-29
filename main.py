import fire
from src.gfpeps import run

def main(
    model: str = 'kitaev_honeycomb_1half',
    Lx: int = 50,
    Ly: int = 50,
    Nv: int = 2,
    max_iter: int = 1000,
    seed: int = 42,
    kernel_params: dict = {},
):
    """Run Gaussian fPEPS ground state optimization.
    
    Args:
        model: Physical model ('kitaev_honeycomb_1half' or 'bcs_pairing_square')
        Lx: Number of k-points in x direction
        Ly: Number of k-points in y direction
        Nv: Number of virtual modes
        max_iter: Maximum optimization iterations
        seed: Random seed
        
        See kernels/
    """    
    return run(
        Lx=Lx,
        Ly=Ly,
        Nv=Nv,
        kernel_names=model,
        kernel_params=kernel_params,
        max_iterations=max_iter,
        seed=seed
    )

if __name__ == '__main__':
    fire.Fire(main)