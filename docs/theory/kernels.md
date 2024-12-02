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

where the constant term is:

$$\mathrm{const.} = \frac{1}{2} \sum_{k,\alpha\beta} \varepsilon_{\alpha\beta}(k)$$

- $\varepsilon_{\alpha\beta}(k) = \varepsilon(k)\delta_{\alpha\beta}$

- $\Delta_{\alpha\beta}(k) = \Delta(k)\delta_{\alpha\uparrow}\delta_{\beta\downarrow}$

- $\varepsilon(k) = -2t(\cos k_x + \cos k_y) - \mu $

- $\Delta(k) = 2(D_x \cos k_x + D_y \cos k_y)$

### Majorana Form
The relation between complex fermion expectation values and the covariance matrix $\Gamma$ is:

$$\langle c_\mu^\dagger c_\nu \rangle = \frac{1}{2} \Big( \delta_{\mu\nu} - \Gamma_{2\mu-1,2\nu} \Big)$$

$$\langle c_\mu^\dagger c_\nu^\dagger \rangle = \frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) + \frac{i}{4}(-\Gamma_{2\mu-1,2\nu-1} + \Gamma_{2\mu,2\nu})$$

$$\langle c_\mu c_\nu \rangle = -\frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) - \frac{i}{4}(\Gamma_{2\mu-1,2\nu-1} - \Gamma_{2\mu,2\nu})$$

For the BCS model with $\alpha=\uparrow = 1$, $\beta=\downarrow=2$, we have:

$$ \langle c_{\uparrow}^\dagger c_\uparrow -\frac{1}{2} \rangle = -\frac{1}{2}\Gamma_{1,2} $$

$$ \langle c_{\downarrow}^\dagger c_\downarrow -\frac{1}{2} \rangle = -\frac{1}{2}\Gamma_{3,4}$$

$$ \langle c_{\uparrow}^\dagger c_\downarrow^\dagger \rangle = \frac{1}{4} (\Gamma_{1,4} + \Gamma_{2,3}) + \frac{i}{4}(-\Gamma_{1,3} + \Gamma_{2,4})$$

and the adjoint of $\langle c_{\uparrow}^\dagger c_\downarrow^\dagger \rangle$ 

$$ \langle c_{\downarrow} c_\uparrow \rangle = -\langle c_{\uparrow} c_\downarrow \rangle = \frac{1}{4} (\Gamma_{1,4} + \Gamma_{2,3}) - \frac{i}{4}(-\Gamma_{1,3} + \Gamma_{2,4})$$

Since $\Delta(k)$ is real, the imaginary terms $\frac{i}{4}(-\Gamma_{1,3} + \Gamma_{2,4})$ cancel in the final Hamiltonian.

The resulting Majorana Hamiltonian in momentum space is:

$$h_{\text{BCS}}(\mathbf{k}) = \frac{1}{4}\begin{pmatrix} 
0 & -\varepsilon(k) & 0 & \Delta(k) \\
\varepsilon(k) & 0 & \Delta(k) & 0 \\
0 & -\Delta(k) & 0 & -\varepsilon(k) \\
-\Delta(k) & 0 & \varepsilon(k) & 0
\end{pmatrix}$$

The equivalent code should be

```python
def bcs_pairing_kernel(k, t=1.0, Dx=0.0, Dy=0.0, mu=0.0):
    # Single-particle dispersion
    eps_k = -2.0 * t * jnp.sum(jnp.cos(k)) - mu
    # k-dependent pairing
    delta_k = 2.0 * (Dx * jnp.cos(k[0]) + Dy * jnp.cos(k[1]))
    
    # Construct 4×4 antisymmetric matrix in Majorana basis
    h_k = 0.25 * jnp.array([
        [0, -eps_k, 0, delta_k],
        [eps_k, 0, delta_k, 0],
        [0, -delta_k, 0, -eps_k],
        [-delta_k, 0, eps_k, 0]
    ])
    
    return h_k
```

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

```python
def kitaev_honeycomb_kernel_s12(k, Jx=1.0, Jy=1.0, Jz=1.0):
    """Construct energy kernel for spin-1/2 Kitaev honeycomb model.
    
    Args:
        k (jnp.ndarray): Momentum vector (kx, ky)
        Jx, Jy, Jz (float): Coupling strengths
        
    Returns:
        jnp.ndarray: Hamiltonian matrix h(k) in Majorana basis, shape (2, 2)
    """
    kx, ky = k[0], k[1]
    Jk = Jz + Jx * jnp.exp(1j * kx) + Jy * jnp.exp(1j * ky)
    h_k = jnp.array([[0, Jk], [-Jk, 0]])
    return h_k
```
#### Result:
- $D=4\ \ , E=-0.1967948$
- $D=8\ \ , E=-0.1968234$
- $D=\infty,E=-0.196824657567299$

### Spin-$\frac{3}{2}$ Majorana Form
For spin-$\frac{3}{2}$ systems, we have four types of Majorana operators at each site:

1. Three primary operators $\tilde{\theta}_i^a$ for $a \in \{x,y,z\}$
2. One composite operator $\tilde{\theta}_i^{xyz}$ 

The Hamiltonian can be written as:

$$H = -\frac{i}{4} \sum_{\langle i j\rangle_a} J_a u_{ij}^a (\theta_i^{xyz} - \tilde{\theta}_i^a)(\theta_j^{xyz} - \tilde{\theta}_j^a)$$

Notice here this formula is just a simple form to indicate the bond dimension of this Hamiltonian. The basic Majorana fermions in this Hamiltonian is still $\theta_i^{x},\theta_i^{y},\theta_i^{z}$ on each site.

Thus, it is not quadratic, if we want to consider mean field theory, we should obtain the effective Hamiltonian.

Its mean field form is

$$\begin{aligned}
H_{\mathrm{MF}}(\{u\})& =-\frac{i}{4}\sum_{\langle ij\rangle_a}J_a u_{ij}^a\tilde{\theta}_i^a\tilde{\theta}_j^a-iD_z\sum_i\theta_i^x\theta_i^y \\
&+\sum_{\langle ij\rangle_a}\frac{iJ_au_{ij}^a}4\left\{\frac{\epsilon_{opq}\epsilon_{rst}}4\langle\theta_i^o\theta_i^p\theta_j^r\theta_j^s\rangle\theta_i^q\theta_j^t\right. \\
&+\frac{\epsilon_{lmn}}2\left(\theta_i^m\theta_i^n\langle\theta_i^l\theta_j^x\theta_j^y\theta_j^z\rangle-\theta_j^m\theta_j^n\langle\theta_j^l\theta_i^x\theta_i^y\theta_i^z\rangle\right) \\
&\left.+\frac{\epsilon_{u\nu w}}2\left[\left(Q_i^{u\nu}\theta_i^w\tilde{\theta}_j^a-\Delta_{ij}^{w\tilde{a}}\theta_i^u\theta_i^\nu+iQ_i^{u\nu}\Delta_{ij}^{w\tilde{a}}\right)-(i\leftrightarrow j)\right]\right\}
\end{aligned}$$

Then we should substitue the following equations into the effective Hamiltonian

$$\langle\theta_i^o\theta_i^p\theta_j^r\theta_j^s\rangle=-Q_i^{op}Q_j^{rs}+\Delta_{ij}^{or}\Delta_{ij}^{ps}-\Delta_{ij}^{os}\Delta_{ij}^{pr}$$

$$\langle\theta_i^l\theta_j^x\theta_j^y\theta_j^z\rangle=\Delta_{ij}^{lx}Q_j^{yz}+\Delta_{ij}^{ly}Q_j^{zx}+\Delta_{ij}^{lz}Q_j^{xy}$$

Then we will get:

$$\begin{aligned}
H_{\mathrm{MF}}(\{u\})& =-\frac{i}{4}\sum_{\langle ij\rangle_a}J_a u_{ij}^a\tilde{\theta}_i^a\tilde{\theta}_j^a-iD_z\sum_i\theta_i^x\theta_i^y \\
&+\sum_{\langle ij\rangle_a}\frac{iJ_au_{ij}^a}4\left\{\frac{\epsilon_{opq}\epsilon_{rst}}4(-Q_i^{op}Q_j^{rs}+\Delta_{ij}^{or}\Delta_{ij}^{ps}-\Delta_{ij}^{os}\Delta_{ij}^{pr})\theta_i^q\theta_j^t\right. \\
&+\frac{\epsilon_{lmn}}2\left(\theta_i^m\theta_i^n(\Delta_{ij}^{lx}Q_j^{yz}+\Delta_{ij}^{ly}Q_j^{zx}+\Delta_{ij}^{lz}Q_j^{xy})-(i\leftrightarrow j)\right) \\
&\left.+\frac{\epsilon_{u\nu w}}2\left[\left(Q_i^{u\nu}\theta_i^w\tilde{\theta}_j^a-\Delta_{ij}^{w\tilde{a}}\theta_i^u\theta_i^\nu+iQ_i^{u\nu}\Delta_{ij}^{w\tilde{a}}\right)-(i\leftrightarrow j)\right]\right\}
\end{aligned}$$

In order to obtain the following form,

$$H_{\mathrm{MF}}(\{u\}=1)=\sum_\mathbf{k}\psi_\mathbf{k}^\dagger H_\mathbf{k}\psi_\mathbf{k},\quad\psi_\mathbf{k}^\dagger=\left(\theta_{A,\mathbf{k}}^x,\theta_{A,\mathbf{k}}^y,\theta_{A,\mathbf{k}}^z,\theta_{B,\mathbf{k}}^x,\theta_{B,\mathbf{k}}^y,\theta_{B,\mathbf{k}}^z\right)$$

We should continue to expand the $H_{\mathrm{MF}}$.

First of all, we can consider the isotropic case, i.e. $D_z=0$ and $Q^{xy}=0$, where all $Q$ reduced to zero.

$$\begin{aligned}
H_{\mathrm{MF}}(\{u\})& =-\frac{i}{4}\sum_{\langle ij\rangle_a}J_a u_{ij}^a\tilde{\theta}_i^a\tilde{\theta}_j^a\\
&+\sum_{\langle ij\rangle_a}\frac{iJ_au_{ij}^a}4\left\{\frac{\epsilon_{opq}\epsilon_{rst}}4(\Delta_{ij}^{or}\Delta_{ij}^{ps}-\Delta_{ij}^{os}\Delta_{ij}^{pr})\theta_i^q\theta_j^t\right. \\
&\left.-\frac{\epsilon_{u\nu w}}2\left[\Delta_{ij}^{w\tilde{a}}\theta_i^u\theta_i^\nu-(i\leftrightarrow j)\right]\right\}
\end{aligned}$$

By Setting $J_a=1$ and $u_{ij}^a=1$

$$\begin{aligned}
H_{\mathrm{MF}}(\{u\})& =\frac{i}{4}\sum_{\langle ij\rangle_a}\left\{-\tilde{\theta}_i^a\tilde{\theta}_j^a +
\frac{\epsilon_{opq}\epsilon_{rst}}4(\Delta_{ij}^{or}\Delta_{ij}^{ps}-\Delta_{ij}^{os}\Delta_{ij}^{pr})\theta_i^q\theta_j^t -\frac{\epsilon_{u\nu w}}2\left[\Delta_{ij}^{w\tilde{a}}\theta_i^u\theta_i^\nu-(i\leftrightarrow j)\right]\right\}
\end{aligned}$$

It is time to consider Fourier transformation.


## Mode Structure

The models have different numbers of degrees of freedom:

### BCS Pairing
- 2 complex fermion modes (↑,↓) per site
- 4 Majorana modes in total
- $h(\mathbf{k})$ dimension: 4×4

### Kitaev Model (Spin-$\frac{1}{2}$)
- 1 complex fermion mode (c⁰) per site
- 2 Majorana modes in total
- $h(\mathbf{k})$ dimension: 2×2

### Kitaev Model (Spin-$\frac{3}{2}$)
- 4 Majorana modes per site ($\tilde{\theta}^x$, $\tilde{\theta}^y$, $\tilde{\theta}^z$, $\tilde{\theta}^{xyz}$)
- $h(\mathbf{k})$ dimension: 4×4
