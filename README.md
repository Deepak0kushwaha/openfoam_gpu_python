# OpenFOAM GPU Python Prototype

This repository demonstrates a proof‑of‑concept for solving partial differential
equations (PDEs) on the GPU using Python.  It provides a simple two‑dimensional
heat equation solver implemented with [CuPy](https://cupy.dev/) that runs
entirely on the GPU.  The solver illustrates how finite‑difference
discretisation can be parallelised across thousands of GPU cores to achieve
significant performance improvements compared with CPU‑only implementations.

## Motivation

OpenFOAM is a widely used open‑source CFD framework written in C++.  While
powerful, running large simulations on CPUs can be time‑consuming.  Recent
research has shown that rewriting CFD algorithms to leverage GPUs can lead to
dramatic speedups.  For example, a single‑GPU immersed‑boundary solver in
Python using Numba and CUDA achieved between **34–54×** speedup over a CPU
solver running on 96–128 cores【225410096998328†L44-L63】.  The acceleration
results primarily from moving the solution of large linear systems to the GPU,
while other parts of the algorithm see more modest gains【225410096998328†L56-L62】.  GPU acceleration for OpenFOAM has similarly focused on coupling
external libraries such as NVIDIA’s AmgX via `amgx4Foam` to offload the
linear solver phase【395347039482267†L485-L498】.

## How it works

The `heat_equation_gpu.py` script solves the 2D heat diffusion equation
∂u/∂t = α∇²u on a uniform grid.  The solver uses CuPy arrays so that
operations such as stencil‑based Laplacian evaluation and time stepping are
executed on the GPU.  Discretised spatial derivatives are parallelised across
GPU cores, following the finite difference strategy where the Laplacian term is
computed using a 5‑point stencil【146289635335462†L228-L274】.  Because all data
resides in GPU memory, there are no slow host‑device transfers during
iteration.  The algorithm employs fixed (Dirichlet) boundary conditions and
outputs the final temperature distribution at the end of the run.

## Getting started

1. **Install CuPy**.  You need a CUDA‑enabled GPU and the appropriate CuPy
   wheel for your CUDA version.  For example:

   ```bash
   pip install cupy-cuda11x  # Replace `11x` with your CUDA major/minor version
   ```

2. **Run the solver**:

   ```bash
   python heat_equation_gpu.py
   ```

   The script initialises a `512×512` grid with a hot square in the centre
   and evolves the temperature field for 1,000 time steps.  At the end
   of the simulation the final field is transferred back to the CPU and
   printed.  You can adjust grid size and step count by passing parameters to
   the `heat_equation` function.

## Extending this work

This repository serves as a starting point for porting more complex CFD
solvers to Python and GPU frameworks:

* **Linear system solvers:** More sophisticated problems require solving large
  sparse systems.  Libraries such as PETSc offer GPU back‑ends, and the
  `amgx4Foam` module connects OpenFOAM to NVIDIA’s AmgX linear solver【395347039482267†L485-L498】.

* **Immersed boundary methods:** GPU implementations of immersed boundary
  solvers written in Python have demonstrated significant speedups by keeping
  all simulation data in GPU memory【225410096998328†L44-L63】.  Adapting these
  techniques to OpenFOAM would involve rewriting key algorithms to avoid
  CPU–GPU communication and optimise memory access patterns.

* **Numba & PyCUDA:** For kernels not expressible via CuPy’s high‑level API,
  consider using [Numba](https://numba.pydata.org/) or [PyCUDA](https://documen.tician.de/pycuda/) to write custom GPU kernels.  These tools
  allow writing Python functions that are just-in-time compiled to run on the
  GPU.

Feel free to modify and extend this prototype.  Pull requests and issues are
welcome!
