from jax import vmap
import jax.numpy as jnp
from .ABD import getGammaProjector
from .Gin import BatchGammaIn,BatchK
from .GaussianLinearMap import GaussianLinearMap

def pairing_amplitude(batch_k, DeltaX, DeltaY, type='d_wave'):
    """
    Calculate pairing amplitude for different BCS pairing types.
    
    Args:
        batch_k: Momentum points
        DeltaX, DeltaY: Pairing amplitudes
        type: Pairing type ('d_wave', 'p_ip_wave', 's_wave')
    """
    kx, ky = batch_k[:, 0], batch_k[:, 1]
    
    if type == 'd_wave':
        # D-wave: cos(kx) - cos(ky)
        return DeltaX * jnp.cos(kx) - DeltaY * jnp.cos(ky)
    elif type == 's_wave':
        # S-wave: constant
        return jnp.ones_like(kx) * DeltaX
    elif type == 'p_ip_wave':
        # P+ip: sin(kx) + i*sin(ky)
        real = DeltaX * jnp.sin(kx) - DeltaY * jnp.sin(ky)
        imag = DeltaX * jnp.sin(ky) + DeltaY * jnp.sin(kx)
        return real + 1j * imag
    else:
        raise ValueError(f"Unknown pairing type: {type}")

def energy_function(hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0, Lx=100, Ly=100, pairing_type='d_wave'):
    """
    Create BCS energy function with specified pairing type.
    
    Args:
        hoping: Hopping parameter
        DeltaX, DeltaY: Pairing amplitudes
        Mu: Chemical potential
        Lx, Ly: System dimensions
        pairing_type: Type of pairing ('d_wave', 'p_ip_wave', 's_wave')
    """
    batch_k = BatchK(Lx, Ly)
    batch_cosk = jnp.sum(jnp.cos(batch_k), axis=1)
    batch_delta = pairing_amplitude(batch_k, DeltaX, DeltaY, pairing_type)

    def energy(BatchGout):
        """BCS energy function"""
        # The following implementations are based on the exact formulas from the user-provided paper,
        # translated into the code's specific basis ordering.
        # Deduced Code Basis: (c_k,↑, c_k,↓, c†_{-k,↑}, c†_{-k,↓})
        # Paper's Basis (assumed): (c_k,↑, c†_{-k,↑}, c_k,↓, c†_{-k,↓})

        G = BatchGout # Use a shorter alias for clarity

        # Density calculation using the paper's formula translated to the code's basis:
        # Paper: n_up = 0.5 - 0.5 * G_paper[0,1]; n_down = 0.5 - 0.5 * G_paper[2,3]
        # Translated: n_up = 0.5 - 0.5 * G_code[0,2]; n_down = 0.5 - 0.5 * G_code[1,3]
        n_up = 0.5 - 0.5 * G[:, 0, 2]
        n_down = 0.5 - 0.5 * G[:, 1, 3]
        rho = n_up + n_down

        # Pairing correlator (kappa) using the paper's formula translated to the code's basis:
        # Paper: κ = 0.25 * [G_paper[0,3] + G_paper[1,2] + i*(G_paper[1,3] - G_paper[0,2])]
        # Translated: κ = 0.25 * [G_code[0,3] + G_code[2,1] + i*(G_code[2,3] - G_code[0,1])]
        real_part = G[:, 0, 3] + G[:, 2, 1]
        imag_part = G[:, 2, 3] - G[:, 0, 1]
        kappa = 0.25 * (real_part + 1j * imag_part)

        # Corrected BCS energy expression using the now-correct physical quantities
        # The kinetic term's coefficient of -2 is restored, based on the paper's definition of ξk.
        # The pairing term's coefficient of 4 is restored, based on the paper's definition of Δk.
        kinetic_energy = -2 * hoping * rho * batch_cosk
        chemical_potential = -Mu * rho
        pairing_energy = 4 * jnp.real(jnp.conj(batch_delta) * kappa)
        
        return jnp.mean(kinetic_energy + chemical_potential + pairing_energy)
    
    return energy

def optimize_runtime_loss(Lx=100, Ly=100, Nv=2, hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0, pairing_type='d_wave'):
    """
    Create optimized runtime loss function.
    
    Args:
        pairing_type: Type of pairing ('d_wave', 'p_ip_wave', 's_wave')
    """
    BatchGin = BatchGammaIn(Lx, Ly, Nv)
    energy = energy_function(hoping=hoping, DeltaX=DeltaX, DeltaY=DeltaY, Mu=Mu, Lx=Lx, Ly=Ly, pairing_type=pairing_type)
    
    def lossT(T):
        """Transform to lossT"""
        Glocal = getGammaProjector(T, Nv)
        BatchGout = GaussianLinearMap(Glocal, BatchGin)
        return jnp.real(energy(BatchGout))
    
    return lossT