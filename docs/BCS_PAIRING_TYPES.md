# BCS Pairing Types in Gaussian fPEPS

This document explains the implementation of different BCS (Bardeen-Cooper-Schrieffer) pairing symmetries in the Gaussian fPEPS framework.

## Overview

The BCS theory describes superconductivity through electron pairing. Different pairing symmetries lead to distinct physical properties:

- **d-wave**: High-Tc cuprates, nodal superconductors
- **p+ip**: Topological superconductors, chiral pairing
- **s-wave**: Conventional superconductors, full gap

## Implementation Philosophy

### Core Principle: Simplicity and Physical Intuition

The implementation follows a simple, physics-driven approach:

1. **Single pairing_amplitude function**: Handles all pairing types with simple if/elif logic
2. **Minimal changes to original structure**: Preserves the elegant original d-wave implementation
3. **String-based pairing type selection**: Simple and intuitive
4. **No complex abstractions**: Avoids unnecessary class hierarchies or factory patterns

### Code Efficiency

**Original approach (d-wave only)**:
- Clean, physics-intuitive implementation
- Perfect for single pairing type

**New approach (multiple pairing types)**:
- Simple function-based extension
- ~20 lines of new code vs ~80 lines in complex version
- Maintains original elegance

## Mathematical Framework

### BCS Hamiltonian

The general BCS Hamiltonian in momentum space is:

```
H_BCS = Σ_k,σ ε_k c_kσ^† c_kσ - Σ_k,k' V_k,k' c_k↑^† c_{-k↓}^† c_{-k'↓} c_k'↑
```

Where:
- `ε_k`: Single-particle dispersion
- `V_k,k'`: Effective attractive interaction
- `c_kσ^†`: Creation operator for electron with momentum k and spin σ

### Mean-Field Approximation

Under mean-field approximation, the pairing amplitude is:

```
Δ_k = Σ_k' V_k,k' ⟨c_{-k'↓} c_k'↑⟩
```

The mean-field Hamiltonian becomes:

```
H_MF = Σ_k [ε_k (c_k↑^† c_k↑ + c_{-k↓}^† c_{-k↓}) + (Δ_k c_k↑^† c_{-k↓}^† + h.c.)]
```

## Implementation Architecture

### Core Energy Function

All pairing types share the same core energy calculation:

```python
def energy(BatchGout):
    """BCS energy function"""
    rhoup = 0.5 + 0.25 * jnp.einsum('ijk,jk->i', BatchGout[:,0:4:2,0:4:2], jnp.array([[0,-1.0],[1.0,0]]))
    rhodn = 0.5 + 0.25 * jnp.einsum('ijk,jk->i', BatchGout[:,1:4:2,1:4:2], jnp.array([[0,-1.0],[1.0,0]]))
    rho = rhoup + rhodn
    kappa = 0.25 * jnp.einsum('ijk,jk->i', BatchGout[:,0:4:2,1:4:2], jnp.array([[0,1.0],[1.0,0]]))
    return jnp.mean(jnp.real(-2 * hoping * rho * batch_cosk + 4 * batch_delta * kappa + Mu * rho))
```

### Pairing Amplitude Function

Simple function with if/elif logic for different pairing types:

```python
def pairing_amplitude(batch_k, DeltaX, DeltaY, type='d_wave'):
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
```

## Pairing Symmetries

### 1. D-Wave Pairing

**Physical System**: High-Tc cuprate superconductors

**Pairing Function**:
```
Δ(k) = Δ_0 (cos(kx) - cos(ky))
```

**Properties**:
- Even function: `Δ(-k) = Δ(k)`
- Nodal structure: gap nodes along diagonal directions
- C4v symmetry
- Conventional (non-topological) superconductor

### 2. P+IP Pairing

**Physical System**: Topological superconductors, chiral p-wave

**Pairing Function**:
```
Δ(k) = Δ_0 (sin(kx) + i*sin(ky))
```

**Properties**:
- Complex function with chiral structure
- No nodes (full gap)
- Chiral pairing (breaks time-reversal symmetry)
- Topological superconductor (supports Majorana fermions)
- **Key feature**: The complex nature gives rise to chiral edge states

**Implementation Details**:
- Real part: `Δ_X * sin(kx) - Δ_Y * sin(ky)`
- Imaginary part: `Δ_X * sin(ky) + Δ_Y * sin(kx)`
- This ensures proper chiral symmetry breaking

### 3. S-Wave Pairing

**Physical System**: Conventional superconductors (e.g., Al, Nb)

**Pairing Function**:
```
Δ(k) = Δ_0 (constant)
```

**Properties**:
- Even function: `Δ(-k) = Δ(k)`
- No nodes (full gap)
- Isotropic pairing
- Conventional superconductor

## Key Differences

| Property | D-Wave | P+IP | S-Wave |
|----------|--------|------|--------|
| **Symmetry** | Even | Complex/Chiral | Even |
| **Nodes** | Yes | No | No |
| **Topology** | Trivial | Non-trivial | Trivial |
| **Typical System** | Cuprates | Chiral p-wave | Conventional |
| **Pairing Function** | cos(kx) - cos(ky) | sin(kx) + i*sin(ky) | constant |
| **Edge States** | No | Yes (Majorana) | No |

## Usage Examples

### Simple Function Calls

```python
# Create energy functions for different pairing types
d_wave_energy = energy_function(
    hoping=1.0, DeltaX=0.5, DeltaY=0.3, pairing_type='d_wave'
)

p_ip_energy = energy_function(
    hoping=1.0, DeltaX=0.5, DeltaY=0.3, pairing_type='p_ip_wave'
)

s_wave_energy = energy_function(
    hoping=1.0, DeltaX=0.5, DeltaY=0.3, pairing_type='s_wave'
)
```

### Optimization

```python
# Create loss functions for optimization
d_wave_loss = optimize_runtime_loss(
    pairing_type='d_wave',
    hoping=1.0, DeltaX=0.5, DeltaY=0.3
)

p_ip_loss = optimize_runtime_loss(
    pairing_type='p_ip_wave',
    hoping=1.0, DeltaX=0.5, DeltaY=0.3
)
```

## Physical Interpretation

### Energy Terms

The energy function contains three main terms:

1. **Hopping Term**: `-2*hoping * rho * batch_cosk`
   - Represents kinetic energy
   - `rho`: Particle density
   - `batch_cosk`: Dispersion relation

2. **Pairing Term**: `4*batch_delta*kappa`
   - Represents superconducting pairing
   - `kappa`: Anomalous density (pairing amplitude)
   - `batch_delta`: Pairing function (depends on symmetry)

3. **Chemical Potential**: `Mu*rho`
   - Controls particle number
   - `rho`: Total particle density

### Symmetry Breaking

- **D-wave**: Breaks rotational symmetry (C4 → C2)
- **P+IP**: Breaks time-reversal symmetry (T → 0) and has chiral structure
- **S-wave**: Preserves all symmetries

### P+IP Specific Physics

The p+ip pairing has unique properties:

1. **Chiral Structure**: The complex pairing amplitude `sin(kx) + i*sin(ky)` creates a chiral order parameter
2. **Topological Protection**: The chiral nature protects edge states
3. **Majorana Fermions**: Supports Majorana zero modes at defects/edges
4. **Time-Reversal Breaking**: The complex nature breaks time-reversal symmetry

## Extending to New Pairing Types

The simple architecture makes it extremely easy to add new pairing types:

### Step 1: Add New Case to pairing_amplitude()
```python
elif type == 'extended_s_wave':
    return DeltaX * jnp.cos(kx) + DeltaY * jnp.cos(ky)
```

### Step 2: Use in energy_function()
```python
energy_function(..., pairing_type='extended_s_wave')
```

**That's it!** No need for complex class structures or factory patterns.

## Backward Compatibility

The original `energy_function` is preserved for backward compatibility:

```python
# Old way (still works, defaults to d-wave)
energy = energy_function(hoping=1.0, DeltaX=0.5)

# New way
energy = energy_function(hoping=1.0, DeltaX=0.5, pairing_type='d_wave')
```

Both produce identical results for d-wave pairing.

## Benefits of the Simple Architecture

1. **Simplicity**: No complex class hierarchies or factory patterns
2. **Physical Intuition**: Maintains the elegant original structure
3. **Easy Extension**: Adding new pairing types requires minimal code
4. **Backward Compatibility**: Original API preserved
5. **Performance**: No overhead from complex abstractions
6. **Maintainability**: Easy to understand and modify

## Momentum Relationship: Continuous vs Lattice Models

### Why sin(k) can replace (kx + i*ky)

In lattice models, we use `sin(kx) + i*sin(ky)` instead of the continuous model's `kx + iky`:

1. **Small Momentum Approximation**: `sin(k) ≈ k` for `k → 0`
2. **Odd Function Property**: `sin(-k) = -sin(k)` ✓
3. **Periodic Boundary Conditions**: `sin(k + 2π) = sin(k)` ✓
4. **Physical Properties Preserved**: Chirality, topology maintained

**Continuous model**: `V_k,k' ∝ (kx + iky)(kx' - iky')`
**Lattice model**: `V_k,k' ∝ (sin(kx) + i*sin(ky))(sin(kx') - i*sin(ky'))`

Both preserve the essential p+ip physics! 