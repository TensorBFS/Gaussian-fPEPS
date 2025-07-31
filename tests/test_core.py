import pytest
import jax
import jax.numpy as jnp
from kitaev import (
    batched_k, batched_Gin, make_correlator, make_loss, 
    kitaev_kernel, KitaevfPEPS, optimize
)

jax.config.update("jax_enable_x64", True)

# Fixed parameters for Kitaev model
NF = 1  # Always 1 for Kitaev spin-1/2

@pytest.fixture(params=[25, 50])
def Lx(request):
    return request.param

@pytest.fixture(params=[25, 50])
def Ly(request):
    return request.param

@pytest.fixture(params=[1, 2])
def Nv(request):
    return request.param

@pytest.fixture
def rng_key():
    return jnp.array([0, 0], dtype=jnp.uint32)

def test_batched_k(Lx, Ly):
    """Test momentum space grid generation."""
    k_points = batched_k(Lx, Ly)
    
    # Check shape
    assert k_points.shape == (Lx * Ly, 2)
    
    # Check values are in correct range
    assert jnp.all(jnp.abs(k_points) <= 2*jnp.pi)

def test_batched_Gin(Lx, Ly, Nv):
    """Test virtual bond matrix generation."""
    Gin = batched_Gin(Lx, Ly, Nv)
    
    # Check shape
    assert Gin.shape == (Lx * Ly, 8 * Nv, 8 * Nv)
    
    # Check matrix properties
    for g in Gin:
        # Check anti-hermiticity
        assert jnp.allclose(g, -jnp.conjugate(g.T))
        # Check trace is zero
        assert jnp.allclose(jnp.trace(g), 0)

def test_kitaev_kernel():
    """Test Kitaev energy kernel."""
    k = jnp.array([0.5, 0.3])
    h_k = kitaev_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0)
    
    # Check shape (2x2 for Kitaev)
    assert h_k.shape == (2, 2)
    
    # Check antisymmetry
    assert jnp.allclose(h_k, -h_k.T)
    
    # Check trace is zero
    assert jnp.allclose(jnp.trace(h_k), 0)

def test_make_correlator(Lx, Ly, Nv, rng_key):
    """Test correlator function creation."""
    correlator = make_correlator(Lx=Lx, Ly=Ly, Nv=Nv)
    
    # Create random orthogonal matrix T
    key1, key2 = jax.random.split(rng_key)
    dim = 2 * NF + 8 * Nv  # 2 + 8*Nv for Kitaev
    T = jnp.array(jax.random.normal(key=key1, shape=(dim, dim)))
    U, S, V = jnp.linalg.svd(T)
    T = U @ V
    
    # Test correlator output
    G = correlator(T)
    
    # Check shape (physical space is 2*Nf = 2 for Kitaev)
    assert G.shape == (Lx*Ly, 2*NF, 2*NF)
    
    # Check matrix properties
    for g in G:
        # Check anti-hermiticity
        assert jnp.allclose(g, -jnp.conjugate(g.T))
        # Check trace is zero
        assert jnp.allclose(jnp.trace(g), 0)

def test_make_loss(Lx, Ly, Nv, rng_key):
    """Test loss function creation."""
    loss_fn = make_loss(Lx=Lx, Ly=Ly, Nv=Nv, Jx=1.0, Jy=1.0, Jz=1.0)
    
    # Create random orthogonal matrix T
    key1, key2 = jax.random.split(rng_key)
    dim = 2 * NF + 8 * Nv
    T = jnp.array(jax.random.normal(key=key1, shape=(dim, dim)))
    U, S, V = jnp.linalg.svd(T)
    T = U @ V
    
    # Test loss function
    energy = loss_fn(T)
    
    # Check if output is real scalar
    assert jnp.issubdtype(energy.dtype, jnp.floating)
    assert energy.shape == ()
    
    # Test parameter dependence
    loss_fn2 = make_loss(Lx=Lx, Ly=Ly, Nv=Nv, Jx=2.0, Jy=1.0, Jz=1.0)
    energy2 = loss_fn2(T)
    assert energy != energy2  # Energy should change with different couplings

def test_KitaevfPEPS():
    """Test KitaevfPEPS class."""
    kitaev = KitaevfPEPS(Lx=10, Ly=10, Nv=1, Jx=1.0, Jy=1.0, Jz=1.0)
    
    # Check initialization
    assert kitaev.Lx == 10
    assert kitaev.Ly == 10
    assert kitaev.Nv == 1
    assert kitaev.T is not None
    assert kitaev.loss is not None
    
    # Check T matrix properties
    dim = 2 + 8 * 1  # 2 + 8*Nv for Kitaev with Nv=1
    assert kitaev.T.shape == (dim, dim)
    
    # Check orthogonality
    assert jnp.allclose(kitaev.T @ kitaev.T.T, jnp.eye(dim), rtol=1e-10)
    
    # Check energy computation
    energy = kitaev.energy()
    assert jnp.issubdtype(energy.dtype, jnp.floating)

def test_optimization_small():
    """Test optimization on small system."""
    kitaev = KitaevfPEPS(Lx=5, Ly=5, Nv=1, Jx=1.0, Jy=1.0, Jz=1.0)
    
    initial_energy = kitaev.energy()
    
    # Run short optimization
    optimized_kitaev, result = optimize(kitaev, max_iterations=5, verbosity=0)
    
    final_energy = optimized_kitaev.energy()
    
    # Energy should decrease (or at least not increase significantly)
    assert final_energy <= initial_energy + 1e-10
    
    # Check that T remains orthogonal
    dim = optimized_kitaev.T.shape[0]
    assert jnp.allclose(
        optimized_kitaev.T @ optimized_kitaev.T.T, 
        jnp.eye(dim), 
        rtol=1e-10
    )