# Macnus
## Magneto-Coriolis Waves Solver
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607675.svg)](https://doi.org/10.5281/zenodo.18607675)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The interaction of rotation, stratification and magnetic fields gives rise to a rich spectrum of waves in geophysical and astrophysical fluids. In this project, carried out as part of research work, we create a framework that includes both viscosity and magnetic diffusivity. The aim is to derive and solve the equations governing **viscous–diffusive Magneto-Coriolis (MC) waves** inside a cylindrical geometry. Using spectral methods (Chebyshev collocation) and a **toroidal–poloidal** decomposition, we construct the eigenvalue problem and compute the corresponding eigenmodes.

The work is organised in two complementary notebooks:

1. **Derivation of equations (SageMath):** symbolic manipulation of the governing equations in cylindrical coordinates, separation into poloidal/toroidal components, and derivation of the final system to be solved.
2. **Numerical solver (Python):** implementation of the spectral method, construction of matrices, eigenvalue computation, and visualisation of the resulting modes.

![](Diffusive-Viscous-MHD/Figures/(m,n)=(2,1)_(nu,eta)=(0.01,0.01).png)

---

## Part 1: Derivation of Poloidal–Toroidal Equations
In the SageMath notebook **MHD-SageMath**, we perform the algebraic steps needed to derive a well-posed eigenvalue problem:

- Decomposition of the velocity and magnetic fields into poloidal and toroidal potentials.  
- Application of the governing equations (hydrodynamic and MHD cases).  
- Obtain a compact matrix system suitable for spectral discretisation.  

This notebook ensures the mathematical consistency of the model before moving to numerical implementation.

---

## Part 2: Numerical Solver for MC Waves
In the Python notebook **Macnus**, we implement the solver step by step:

- Construction of Chebyshev differentiation matrices and boundary conditions.  
- Assembly of the eigenvalue problem matrices based on the derived equations.  
- Use of linear algebra routines to compute eigenfrequencies and eigenmodes.  
- Visualisation of eigenfunctions (radial profiles, meridional slices, convergence analysis).  

The notebook is organised so that each cell can be executed sequentially, from initialisation of parameters to post-processing and plotting.

---

## Main Results
- Computation of eigenfrequencies of Magneto-Coriolis modes in cylindrical geometry.  
- Visualisation of eigenmodes (velocity and magnetic components).  
- Exploration of convergence with respect to the number of collocation points.  
- Framework that can be extended to other boundary conditions or parameter regimes.  

---

## Conclusion
This project demonstrates how symbolic derivation and numerical computation can be combined to tackle complex wave problems in rotating magnetised fluids. The methodology provides a flexible basis for exploring parameter dependence and physical regimes relevant to geophysics and astrophysics.

---

## Use
Run the notebooks in order:

- **Derivation of equations:** [MHD-SageMath](Pol_Tor_Hydro_&_MHD.ipynb)  
- **Numerical solver:** [Macnus](Diffusive-Viscous-MHD/Macnus_SLEPc.ipynb)  

Both require standard scientific Python libraries (`numpy`, `scipy`, `matplotlib`, `mpmath`) and, for the derivation notebook, SageMath. 

## Citation

If you use this software, please cite it as:

> Alexandre Nuyt. (2025). Alnuyt/Magneto-Coriolis-Waves-Solver: Spectral-MC-Solver (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17792073
