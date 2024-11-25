# Gaussian Fermionic PEPS Correlators

## Mathematical Foundation


### Momentum Space Structure
The implementation uses momentum space with mixed boundary conditions:

- Anti-periodic boundary conditions in x: $k_x \in [-\pi + \pi/(2L_x), \pi - \pi/(2L_x)]$
- Periodic boundary conditions in y: $k_y \in [0, 2\pi)$

### Gaussian Mapping
The core correlation computation implements the Gaussian mapping formula:

$$\Gamma_{out} = A + B(D + \Gamma_{in})^{-1}B^T$$

where:

- $\Gamma_{in}(k)$ is the input state at momentum $k$
- $\begin{pmatrix} A & B \\ B^T & D \end{pmatrix}$ is the block structure of $G_{local} = T^\dagger JT$
- Where $J$ is the symplectic form: $\bigoplus_i\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$

## Mode Structure
The implementation handles:

- $N_f$ physical fermion modes per site
- $N_v$ virtual modes per bond
- Total of $4N_v$ virtual modes per site (up, down, left, right)

### Dimension Counting

- Physical space: $2N_f$ Majorana modes
- Virtual space: $8N_v$ Majorana modes
- Total dimension of $\Gamma$: $(N_f + 4N_v) \times (N_f + 4N_v)$