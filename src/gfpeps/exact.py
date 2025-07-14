import jax.numpy as jnp
from jax.scipy.linalg import eigh
import jax
from .Gin import BatchK

def pairing_amplitude(k, pairing_type, DeltaX, DeltaY):
    """统一的 pairing amplitude 生成函数"""
    if pairing_type == "d_wave":
        return DeltaX * jnp.cos(k[0]) - DeltaY * jnp.cos(k[1])
    elif pairing_type == "s_wave":
        return DeltaX
    elif pairing_type == "p_ip_wave":
        real = DeltaX * jnp.sin(k[0]) - DeltaY * jnp.sin(k[1])
        imag = DeltaX * jnp.sin(k[1]) + DeltaY * jnp.sin(k[0])
        return real + 1j * imag
    else:
        raise ValueError(f"Unknown pairing_type: {pairing_type}")

def exact(k, ht, DeltaX, DeltaY, Mu, pairing_type):
    """统一的 exact energy 计算函数"""
    D = pairing_amplitude(k, pairing_type, DeltaX, DeltaY)
    c = jnp.cos(k[0]) + jnp.cos(k[1])
    t = jnp.array([[-ht * c, 0], [0, -ht * c + Mu]])
    # d 的定义应该更直接
    d = jnp.array([[0, D], [-D, 0]])
    # M 的构造必须保证厄米性
    M = jnp.block([[t, d], [d.conj().T, -t]])
    w, _ = eigh(M)
    N = w.shape[0] // 2
    return jnp.sum(w[:N]) + jnp.sum(jnp.diag(t))

def eg(Lx, Ly, ht, DeltaX, DeltaY, Mu, pairing_type='d_wave'):
    """统一的 exact ground state energy 计算，并带有详细的调试输出"""
    KSet = BatchK(Lx, Ly)
    
    print(f"\n--- Debugging Eg Calculation for pairing_type='{pairing_type}' ---")
    print(f"Parameters: Lx={Lx}, Ly={Ly}, ht={ht}, DeltaX={DeltaX}, DeltaY={DeltaY}, Mu={Mu}")
    print(f"Total k-points: {len(KSet)}")
    
    all_energies = []
    
    for i, k in enumerate(KSet):
        # --- Calculation for a single k ---
        D = pairing_amplitude(k, pairing_type, DeltaX, DeltaY)
        c = jnp.cos(k[0]) + jnp.cos(k[1])
        t = jnp.array([[-ht * c, 0], [0, -ht * c + Mu]])
        d = jnp.array([[0, D], [-D, 0]])
        M = jnp.block([[t, d], [d.conj().T, -t]])
        w, _ = eigh(M)
        N = w.shape[0] // 2
        energy_k = jnp.sum(w[:N]) + jnp.sum(jnp.diag(t))
        all_energies.append(energy_k)

    # Calculate final result from the loop
    final_eg = jnp.sum(jnp.array(all_energies)) / KSet.shape[0]

    print(f"\n--- Calculation Summary ---")
    print(f"Final Eg (from debug loop):   {final_eg:.8f}")
    print(f"--- End of Debug ---")

    return final_eg

# 保持向后兼容的函数
def exact_d_wave(k, ht, DeltaX, DeltaY, Mu):
    return exact(k, ht, DeltaX, DeltaY, Mu, 'd_wave')

def exact_s_wave(k, ht, DeltaX, DeltaY, Mu):
    return exact(k, ht, DeltaX, DeltaY, Mu, 's_wave')

def exact_p_ip_wave(k, ht, DeltaX, DeltaY, Mu):
    return exact(k, ht, DeltaX, DeltaY, Mu, 'p_ip_wave')