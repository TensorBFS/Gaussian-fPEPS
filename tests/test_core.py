import pytest
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from src.gfpeps.core.Gin import batched_k, batched_Gin
from src.gfpeps.core.correlator import make_correlator
from src.gfpeps.core.loss import make_loss

jax.config.update("jax_enable_x64", True)

@pytest.fixture(params=[50])
def Lx(request):
    return request.param

@pytest.fixture(params=[50])
def Ly(request):
    return request.param

@pytest.fixture(params=[2, 3])
def Nf(request):
    return request.param

@pytest.fixture(params=[2, 4])
def Nv(request):
    return request.param

@pytest.fixture
def rng_key():
    return jnp.array([0, 0], dtype=jnp.uint32)

def test_batched_k(Lx, Ly):
    # Test with parameterized grid size
    k_points = batched_k(Lx, Ly)
    
    # Check shape
    assert k_points.shape == (Lx * Ly, 2)
    
    # Check values are in correct range [-π, π]
    assert jnp.all(jnp.abs(k_points) <= 2*jnp.pi)

def test_batched_Gin(Lx, Ly, Nv):
    Gin = batched_Gin(Lx, Ly, Nv)
    
    # Check shape
    assert Gin.shape == (Lx * Ly, 8 * Nv, 8 * Nv)
    
    # Check matrix properties
    for g in Gin:
        # Check anti-hermiticity
        assert jnp.allclose(g, -jnp.conjugate(g.T))
        # check trace is zero
        assert jnp.allclose(jnp.trace(g), 0)

def test_make_correlator(Lx, Ly, Nf, Nv, rng_key):
    correlator = make_correlator(Lx=Lx, Ly=Ly, Nf=Nf, Nv=Nv)
    
    # Create a random tensor for testing
    key1, key2 = jax.random.split(rng_key)

    # Create a random unitary matrix T:
    T = jnp.array(jax.random.normal(key=key1, shape=(2*Nf + 8*Nv, 2*Nf + 8*Nv)))
    U, S, V = jnp.linalg.svd(T)
    T = U @ V
    
    # Test correlator output
    G = correlator(T)
    
    # Check shape
    assert G.shape == (Lx*Ly, 2*Nf, 2*Nf)
    
    # Check matrix properties
    for g in G:
        # Check anti-hermiticity
        assert jnp.allclose(g, -jnp.conjugate(g.T))
        # check trace is zero
        assert jnp.allclose(jnp.trace(g), 0)

def test_make_loss(Lx, Ly, Nf, Nv, rng_key):
    # Define a simple test kernel function
    def test_kernel(k, coupling=1.0):
        # the return matrix of a kernel function should be 2*Nf x 2*Nf
        return coupling * jnp.ones((2*Nf, 2*Nf))
    
    loss_fn = make_loss(test_kernel, Lx=Lx, Ly=Ly, Nf=Nf, Nv=Nv, coupling=1.0)
    
    # Create a random tensor for testing
    key1, key2 = jax.random.split(rng_key)

    # Create a random unitary matrix T:
    T = jnp.array(jax.random.normal(key=key1, shape=(2*Nf + 8*Nv, 2*Nf + 8*Nv)))
    U, S, V = jnp.linalg.svd(T)
    T = U @ V
    
    # Test loss function output
    loss = loss_fn(T)
    
    # Check if output is real
    print(loss)
    assert jnp.issubdtype(loss.dtype, jnp.floating)
    
    # Check if loss changes with different coupling
    loss_fn2 = make_loss(test_kernel, Lx=Lx, Ly=Ly, Nf=Nf, Nv=Nv, coupling=2.0)
    loss2 = loss_fn2(T)
    assert loss != loss2  # Loss should change with different coupling