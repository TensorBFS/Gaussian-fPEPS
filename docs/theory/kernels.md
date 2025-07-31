# Kitaev Honeycomb Spin-1/2 Model

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

## Kitaev Honeycomb Model

### Spin-$\frac{1}{2}$ Majorana Form
Using Kitaev's Majorana representation $\sigma_j^d = i\gamma_j^d\gamma_j^0$, the effective Hamiltonian becomes:

$$H_{\text{eff}} = i \sum_{d} \sum_{\langle j k\rangle \in d} J_{d} u_{jk} \gamma_{j}^{0} \gamma_{k}^{0}$$

where $u_{jk} = i\gamma_j^d\gamma_k^d$ is the Z₂ gauge field. 

The honeycomb lattice can be decomposed into a Bravais lattice with a two-site (A,B) unit cell. With the fixed gauge choice $u_{jk}=+1$, the real-space Hamiltonian becomes:

$$H = i\sum_{i,j}\left[J_z\gamma_{i,j}^A\gamma_{i,j}^B - J_x\gamma_{i,j}^B\gamma_{i,j+1}^A - J_y\gamma_{i,j}^B\gamma_{i+1,j}^A\right]$$

where $(i,j)$ labels the unit cell position.

To transform to momentum space, we use the Fourier transform:

$$\gamma^{A/B}_{\mathbf{r}} = \frac{1}{\sqrt{N}}\sum_{\mathbf{k}}\exp(i\mathbf{k}\cdot\mathbf{r})\gamma^{A/B}_{\mathbf{k}}$$

This yields:

$$ H = i\sum_{\mathbf{k}}\left[J_z - J_x e^{i\mathbf{k}\cdot\mathbf{\hat{x}}} - J_y e^{i\mathbf{k}\cdot\mathbf{\hat{y}}}\right]\gamma^A_{\mathbf{k}}\gamma^B_{-\mathbf{k}}$$

The Majorana Hamiltonian in momentum space then takes the form:

$$h_{\text{Kitaev}}^{(1/2)}(\mathbf{k}) = \begin{pmatrix} 
0 & J(\mathbf{k}) \\ 
-J(\mathbf{k}) & 0
\end{pmatrix}$$

with:

$$J(\mathbf{k}) = J_z - J_x e^{i\mathbf{k}\cdot\hat{x}} - J_y e^{i\mathbf{k}\cdot\hat{y}}$$

### Implementation

```python
def kitaev_kernel(k, Jx=1.0, Jy=1.0, Jz=1.0):
    """Construct energy kernel for spin-1/2 Kitaev honeycomb model.
    
    Args:
        k: Momentum vector (kx, ky)
        Jx, Jy, Jz: Coupling strengths
        
    Returns:
        2x2 antisymmetric matrix h(k) in Majorana basis
    """
    kx, ky = k[0], k[1]
    Jk = Jz - Jx * jnp.exp(1j * kx) - Jy * jnp.exp(1j * ky)
    return jnp.array([[0, Jk], [-Jk, 0]]) / 4.0
```

### Energy Results
Benchmark ground state energies for different bond dimensions:
- $D=4$: $E=-0.1967948$
- $D=8$: $E=-0.1968234$  
- $D=\infty$: $E=-0.196824657567299$

## Mode Structure

The Kitaev spin-1/2 model has:
- 1 complex fermion mode (c⁰) per site
- 2 Majorana modes in total
- $h(\mathbf{k})$ dimension: 2×2
- Fixed $N_f = 1$ (number of physical fermion flavors)
