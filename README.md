# meep_adjoint

A lightweight version of the Adjoint-solver module for MEEP originally developed by Homer Reid.

## Install

The current module uses `fenics` to implement the basis function expansion. To install both `fenics` and `pymeep` in the same `conda` environment, run

```bash
conda create -n adjoint -c conda-forge python=3.7 mpich fenics pymeep
```

To install a development version of `meep_adjoint` within your new `conda` environment, simply run

```bash
pip install -e .
```

in the project's home directory.

## Planned features and tasks

- [x] Validate method with finite difference approximation
- [ ] Enable 3D implementation
- [ ] Enable broadband objective functions
- [x] Demonstrate nonlinear filtering algorithms of design variables
- [ ] Demonstrate constraints
- [x] Streamline objective function parsing

## Original Documentation

http://homerreid.github.io/meep-adjoint-documentation
