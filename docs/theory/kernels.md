# Energy Kernels in Gaussian-fPEPS

## Mathematical Foundation

### Majorana Hamiltonian
In Majorana representation, any quadratic Hamiltonian takes the form:

$$H = \frac{i}{2} \sum_{\mathbf{k}} \gamma_{\mathbf{k}}^T h(\mathbf{k}) \gamma_{-\mathbf{k}}$$

where:

- $\gamma_{\mathbf{k}}$ are Majorana operators in momentum space

- $h(\mathbf{k})$ is an antisymmetric matrix: $h^T = -h$

### Energy Expectation
Given a Gaussian state with covariance matrix $\Gamma_{ij}(\mathbf{k}) = \frac{i}{2}\langle[\gamma_i(\mathbf{k}), \gamma_j(-\mathbf{k})]\rangle$, the energy expectation is:

$$\langle H \rangle = \sum_{\mathbf{k}}\sum_{ij} \Gamma_{ij}(\mathbf{k}) h_{ij}(\mathbf{k})$$

## BCS Pairing Model

### Complex Fermion Form
The BCS pairing Hamiltonian in complex fermion basis:

$$H = \mathrm{const.} + \sum_{k,\alpha\beta} \varepsilon_{\alpha\beta}(k) (c^\dagger_{k\alpha}c_{k\beta} -\frac{1}{2})+ [\Delta_{\alpha\beta}(k)c_{k\alpha}c_{-k\beta} + h.c.]$$

$$\mathrm{const} = \frac{1}{2} \sum_{k,\alpha\beta} \varepsilon_{\alpha\beta}(k) $$

where:

- $\varepsilon_{\alpha\beta}(k) = \varepsilon(k)\delta_{\alpha\beta}$

- $\Delta_{\alpha\beta}(k) = \Delta(k)\delta_{\alpha\uparrow}\delta_{\beta\downarrow}$

- $\varepsilon(k) = -2t(\cos k_x + \cos k_y) - \mu $

- $\Delta(k) = 2(D_x \cos k_x + D_y \cos k_y)$

### Majorana Form
Using the transformation between complex fermions and Majorana operators:

$$\langle c_\mu^\dagger c_\nu \rangle = \frac{1}{2} \Big( \delta_{\mu\nu} - \Gamma_{2\mu-1,2\nu} \Big)$$

$$\langle c_\mu^\dagger c_\nu^\dagger \rangle = \frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) + \frac{i}{4}(-\Gamma_{2\mu-1,2\nu-1} + \Gamma_{2\mu,2\nu})$$

Thus, further, $\alpha=\uparrow = 1$, $\beta=\downarrow=2$, we obtain

$$ \langle c_{\uparrow}^\dagger c_\uparrow -\frac{1}{2} \rangle = -\frac{1}{2}\Gamma_{1,2} $$

$$ \langle c_{\downarrow}^\dagger c_\downarrow -\frac{1}{2}  \rangle = -\frac{1}{2}\Gamma_{3,4}$$

$$ \langle c_{\uparrow}^\dagger c_\downarrow^\dagger \rangle = \frac{1}{4} (\Gamma_{1,4} + \Gamma_{2,3}) + \frac{i}{4}(-\Gamma_{1,3} + \Gamma_{2,4})$$

We can obtain the Majorana Hamiltonian in momentum space. 

$$h_{\text{BCS}}(\mathbf{k}) = \frac{1}{4}\begin{pmatrix} 
0 & -\varepsilon(k) & -i\Delta^*(k) & \Delta^*(k) \\
\varepsilon(k) & 0 & \Delta^*(k) & i\Delta^*(k) \\
-\Delta(k) & 0 & 0 & -\varepsilon(k) \\
0 & -\Delta(k) & \varepsilon(k) & 0
\end{pmatrix}$$

where:
- The diagonal blocks correspond to normal terms ($c^\dagger_{k\alpha}c_{k\alpha}$)
- The off-diagonal blocks correspond to pairing terms ($c_{k\alpha}c_{-k\beta}$ and h.c.)
- The antisymmetry of $h(\mathbf{k})$ is manifest
- The structure preserves the relation between $\langle c_{\uparrow} c_\downarrow \rangle$ and $\Gamma_{1,4}, \Gamma_{3,2}, \Gamma_{1,3}, \Gamma_{2,4}$

## Kitaev Honeycomb Model

### Spin Form
Original spin Hamiltonian on honeycomb lattice:

$$H = -\sum_{d} \sum_{\langle j k\rangle \in d} J_{d} \sigma_{j}^{d} \sigma_{k}^{d}$$

where:
- $d \in \{x,y,z\}$ labels bond directions
- $\sigma_j^d$ are Pauli matrices at site $j$

### Majorana Form
Using Kitaev's Majorana representation $\sigma_j^d = ic_j^dc_j^0$:

$$H_{\text{eff}} = i \sum_{d} \sum_{\langle j k\rangle \in d} J_{d} u_{jk} c_{j}^{0} c_{k}^{0}$$

where $u_{jk} = ic_j^dc_k^d$ is the Z₂ gauge field.

In momentum space with fixed gauge $u_{jk}=1$:

$$h_{\text{Kitaev}}(\mathbf{k}) = \begin{pmatrix} 
0 & J(\mathbf{k}) \\ 
-J(\mathbf{k}) & 0
\end{pmatrix}$$

where:

$$J(\mathbf{k}) = J_z - J_x e^{i\mathbf{k}\cdot\hat{x}} - J_y e^{i\mathbf{k}\cdot\hat{y}}$$

## Mode Structure

For each model:

### BCS Pairing
- 2 complex fermion modes (↑,↓) per site
- 4 Majorana modes in total
- $h(\mathbf{k})$ dimension: 4×4

### Kitaev Model
- 1 complex fermion mode (c⁰) per site
- 2 Majorana modes in total
- $h(\mathbf{k})$ dimension: 2×2

```python
def kernel(k, **params):
    """
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        **params: Model-specific parameters
    
    Returns:
        jnp.ndarray: Hamiltonian matrix h(k) in Majorana basis
    """
```

```python
def bcs_pairing_kernel(k, t=1.0, Dx=0.0, Dy=0.0, mu=0.0):
    # Single-particle dispersion
    eps_k = -2 * t * jnp.sum(jnp.cos(k))
    # k-dependent pairing
    delta_k = 2 * (Dx * jnp.cos(k[0]) + Dy * jnp.cos(k[1]))
    
    # Construct 4×4 antisymmetric matrix in Majorana basis
    h_k = jnp.array([
        [0, eps_k - mu, delta_k, 0],
        [-(eps_k - mu), 0, 0, delta_k],
        [-delta_k, 0, 0, eps_k - mu],
        [0, -delta_k, -(eps_k - mu), 0]
    ])
    
    return h_k
```

```python
def kitaev_honeycomb_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    # Use distorted Bravais lattice (square) for computation
    kx, ky = k[0], k[1]
    
    # J(k) = Jz - Jx exp(ikx) - Jy exp(iky)
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    
    # Construct antisymmetric matrix
    h_k = jnp.array([[0, Jk], [-Jk, 0]])
    
    return h_k