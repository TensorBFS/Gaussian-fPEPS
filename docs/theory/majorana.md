# Majorana Representation

## Basic Properties of Majorana Operators
Majorana operators satisfy the anticommutation relation:

$$\{\gamma_{\mu}, \gamma_{\nu}\} = 2\delta_{\mu\nu}$$

### Covariance Matrix
In the Gaussian formalism, a fermionic state is completely characterized by its covariance matrix $\Gamma$. For Majorana operators $\gamma_i$, the covariance matrix elements are:

$$\Gamma_{\mu\nu} = \frac{i}{2}\langle[\gamma_\mu, \gamma_\nu]\rangle$$

This matrix satisfies:

- $\Gamma^2 = 1$ (Fermionic constraint)

- $\Gamma^T = -\Gamma$ (Antisymmetry)

Thus,

$$\gamma_{\mu}\gamma_{\nu}=\delta_{\mu\nu} - i\Gamma_{\mu,\nu}$$

## Complex-Majorana Transformation
The relation between complex fermion operators and Majorana operators is:

$$\gamma_{2\mu-1} = c_\mu^\dagger + c_\mu, \quad \gamma_{2\mu} = -i(c_\mu^\dagger - c_\mu)$$

Inversely:

$$c_\mu = \frac{1}{2}(\gamma_{2\mu-1} - i\gamma_{2\mu}), \quad c_\mu^\dagger = \frac{1}{2}(\gamma_{2\mu-1} + i\gamma_{2\mu})$$

## Expectation Values in Terms of $\Gamma$
The expectation values $\langle c_\mu^\dagger c_\nu \rangle$ and $\langle c_\mu c_\nu \rangle$ can be expressed using the covariance matrix $\Gamma$. 

### **1. Deriving $\langle c_\mu^\dagger c_\nu \rangle$**
Using the definition:

$$\begin{aligned}
\langle c_\mu^\dagger c_\nu \rangle &= \left\langle \frac{\gamma_{2\mu-1} + i\gamma_{2\mu}}{2} \cdot \frac{\gamma_{2\nu-1} - i\gamma_{2\nu}}{2} \right\rangle \\
&= \frac{1}{4} \Big( \langle \gamma_{2\mu-1} \gamma_{2\nu-1} \rangle - i\langle \gamma_{2\mu-1} \gamma_{2\nu} \rangle + i\langle \gamma_{2\mu} \gamma_{2\nu-1} \rangle + \langle \gamma_{2\mu} \gamma_{2\nu} \rangle \Big)
\end{aligned}$$

Substituting $\langle \gamma_i \gamma_j \rangle = \delta_{ij} - i\Gamma_{ij}$:

$$\begin{aligned}
\langle c_\mu^\dagger c_\nu \rangle &= \frac{1}{4} \Big( (\delta_{2\mu-1,2\nu-1} - i\Gamma_{2\mu-1,2\nu-1}) \\
&\quad\quad - i(\delta_{2\mu-1,2\nu} - i\Gamma_{2\mu-1,2\nu}) \\
&\quad\quad + i(\delta_{2\mu,2\nu-1} - i\Gamma_{2\mu,2\nu-1}) \\
&\quad\quad + (\delta_{2\mu,2\nu} - i\Gamma_{2\mu,2\nu}) \Big)
\end{aligned}$$

Simplifying using $\delta_{ij}$ and the antisymmetry of $\Gamma$:

$$\langle c_\mu^\dagger c_\nu \rangle = \frac{1}{2} \Big( \delta_{\mu\nu} - \Gamma_{2\mu-1,2\nu} \Big)$$

### **2. Deriving $\langle c_\mu^\dagger c_\nu^\dagger \rangle$**
Using the definition:

$$\begin{aligned}
\langle c^\dagger_\mu c^\dagger_\nu \rangle &= \left\langle \frac{\gamma_{2\mu-1} + i\gamma_{2\mu}}{2} \cdot \frac{\gamma_{2\nu-1} + i\gamma_{2\nu}}{2} \right\rangle \\
&= \frac{1}{4} \Big( \langle \gamma_{2\mu-1} \gamma_{2\nu-1} \rangle + i\langle \gamma_{2\mu-1} \gamma_{2\nu} \rangle + i\langle \gamma_{2\mu} \gamma_{2\nu-1} \rangle - \langle \gamma_{2\mu} \gamma_{2\nu} \rangle \Big)
\end{aligned}$$

Substituting $\langle \gamma_i \gamma_j \rangle = \delta_{ij} - i\Gamma_{ij}$:

$$\begin{aligned}
\langle c_\mu^\dagger c_\nu^\dagger \rangle &= \frac{1}{4} \Big( (\delta_{2\mu-1,2\nu-1} - i\Gamma_{2\mu-1,2\nu-1}) \\
&\quad\quad + i(\delta_{2\mu-1,2\nu} - i\Gamma_{2\mu-1,2\nu}) \\
&\quad\quad + i(\delta_{2\mu,2\nu-1} - i\Gamma_{2\mu,2\nu-1}) \\
&\quad\quad - (\delta_{2\mu,2\nu} - i\Gamma_{2\mu,2\nu}) \Big)
\end{aligned}$$

Simplifying:

$$\langle c_\mu^\dagger c_\nu^\dagger \rangle = \frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) + \frac{i}{4}(-\Gamma_{2\mu-1,2\nu-1} + \Gamma_{2\mu,2\nu})$$

You can find the same formula in our paper, just the following paragraph of Eq.(11).


### **3. Deriving $\langle c_\mu c_\nu \rangle$**
Using the definition:

$$\begin{aligned}
\langle c_\mu c_\nu \rangle &= \left\langle \frac{\gamma_{2\mu-1} - i\gamma_{2\mu}}{2} \cdot \frac{\gamma_{2\nu-1} - i\gamma_{2\nu}}{2} \right\rangle \\
&= \frac{1}{4} \Big( \langle \gamma_{2\mu-1} \gamma_{2\nu-1} \rangle - i\langle \gamma_{2\mu-1} \gamma_{2\nu} \rangle - i\langle \gamma_{2\mu} \gamma_{2\nu-1} \rangle - \langle \gamma_{2\mu} \gamma_{2\nu} \rangle \Big)
\end{aligned}$$

Substituting $\langle \gamma_i \gamma_j \rangle = \delta_{ij} - i\Gamma_{ij}$:

$$\begin{aligned}
\langle c_\mu c_\nu \rangle &= \frac{1}{4} \Big( (\delta_{2\mu-1,2\nu-1} - i\Gamma_{2\mu-1,2\nu-1}) \\
&\quad\quad - i(\delta_{2\mu-1,2\nu} - i\Gamma_{2\mu-1,2\nu}) \\
&\quad\quad - i(\delta_{2\mu,2\nu-1} - i\Gamma_{2\mu,2\nu-1}) \\
&\quad\quad - (\delta_{2\mu,2\nu} - i\Gamma_{2\mu,2\nu}) \Big)
\end{aligned}$$

Simplifying:

$$\langle c_\mu c_\nu \rangle = -\frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) - \frac{i}{4}(\Gamma_{2\mu-1,2\nu-1} - \Gamma_{2\mu,2\nu})$$

### Final Result

$$\langle c_\mu^\dagger c_\nu \rangle = \frac{1}{2} \Big( \delta_{\mu\nu} - \Gamma_{2\mu-1,2\nu} \Big)$$

$$\langle c_\mu^\dagger c_\nu^\dagger \rangle = +\frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) + \frac{i}{4}(-\Gamma_{2\mu-1,2\nu-1} + \Gamma_{2\mu,2\nu})$$

$$\langle c_\mu c_\nu \rangle = -\frac{1}{4} (\Gamma_{2\mu-1,2\nu} + \Gamma_{2\mu,2\nu-1}) + \frac{i}{4}(-\Gamma_{2\mu-1,2\nu-1} + \Gamma_{2\mu,2\nu})$$