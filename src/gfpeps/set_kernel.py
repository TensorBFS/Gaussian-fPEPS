# a helper function to set kernel functions for different models

def set_kernel(kernel_name: str):
    """A helper function to set up the appropriate kernel function and number of fermion flavors for different models.
    
    This function serves as a central registry for different kernel functions used in the Gaussian fPEPS calculations.
    Each kernel corresponds to a specific physical model and determines the number of fermion flavors (Nf) required.
    
    Available kernels(you can add more kernels in `gfpeps/kernels`):
        - 'bcs_pairing_square': BCS pairing on square lattice (Nf=2)
        - 'kitaev_honeycomb_1half': Kitaev model on honeycomb lattice (Nf=1)
    
    Args:
        kernel_name (str): Name of the kernel to use. Must be one of the available kernels.
    
    Returns:
        tuple: A tuple containing:
            - callable: The kernel function that takes momentum k and additional parameters
            - int: Number of fermion flavors (Nf) required for this kernel
    
    Raises:
        ValueError: If kernel_name is not recognized
    """
    if kernel_name == 'bcs_pairing_square':
        from .kernels.bcs_paring_square import bcs_pairing_square_kernel
        return bcs_pairing_square_kernel, 2
    elif kernel_name == 'kitaev_honeycomb_1half':
        from .kernels.kitaev_honeycomb_1half import kitaev_honeycomb_1half_kernel
        return kitaev_honeycomb_1half_kernel, 1
    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}. Available kernels: ['bcs_pairing_square', 'kitaev_honeycomb_1half']")