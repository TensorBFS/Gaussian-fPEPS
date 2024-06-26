# Gaussian-fPEPS

Translate the projected BCS state to the fermionic projected entangled pair state(fPEPS).

# About

Demo code for the paper [Projected d-wave superconducting state: a fermionic projected entangled pair state study](https://arxiv.org/abs/2208.04566).


# Example

- Using [Pluto](https://github.com/fonsp/Pluto.jl) Notebook and open example.jl
- An alternative implementation of "translation" is `translate.py`

# Install

Dependencies are maintained via `hatch`. Run codes with
```
  hatch run python gfpeps_app.py
  hatch run python translate.py
```

## Python dependencies:

[JAX](https://github.com/google/jax), [PyManopt](https://pymanopt.org/) and [h5py](https://docs.h5py.org/en/stable/) are required to run gfpeps/main.py.

## Julia dependencies:

[Pluto](https://github.com/fonsp/Pluto.jl) is needed to open the example notebook to open example.jl.
