import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from loss import optimize_runtime_loss

lossR = jit(optimize_runtime_loss(Nv=2))

if __name__ == '__main__':
# TEMP:
    Nv = 2
    R = jnp.array(np.random.rand(20,20))
    print(lossR(R))