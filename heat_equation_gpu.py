"""
GPU‑accelerated 2D heat equation solver using CuPy.

This script implements a simple finite‑difference solver for the two‑dimensional
heat diffusion equation ∂u/∂t = α ∇²u on a regular Cartesian grid. The
computation is performed entirely on the GPU using the CuPy library. By
running the numerical kernel on the GPU, the expensive Laplacian computation
and time integration benefit from the massive parallelism available on
graphics processors.  The approach follows the strategy described in
scientific literature, where spatial derivative calculations are parallelised
across GPU cores and the Laplacian term is computed using stencil
operations【146289635335462†L228-L274】.

The solver initialises a temperature field on a 2D domain with a hot
square in the centre, then iteratively applies the diffusion operator for
a specified number of time steps.  Boundary conditions are fixed (Dirichlet)
and remain at the initial temperature values.  After the simulation, the
resulting temperature field is returned to the host and printed.

Requirements:
    - A CUDA‑capable GPU
    - CuPy installed with a version matching your CUDA toolkit

Example:
    python heat_equation_gpu.py

Author: OpenAI Assistant
License: MIT
"""

import cupy as cp


def heat_equation(nx: int = 512,
                  ny: int = 512,
                  steps: int = 1000,
                  dx: float = 0.01,
                  dt: float = 1e-4,
                  alpha: float = 1.0) -> cp.ndarray:
    """Solve the 2D heat equation on a GPU.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    steps : int
        Number of time steps to integrate.
    dx : float
        Spatial resolution (grid spacing). The same spacing is used in both
        directions for simplicity.
    dt : float
        Time step size.
    alpha : float
        Thermal diffusivity constant.

    Returns
    -------
    cp.ndarray
        Final temperature field stored in GPU memory.
    """
    # Allocate the temperature field on the GPU and set initial conditions.
    u = cp.zeros((nx, ny), dtype=cp.float32)
    # Define a hot region in the centre of the domain
    x_start, x_end = nx // 4, 3 * nx // 4
    y_start, y_end = ny // 4, 3 * ny // 4
    u[x_start:x_end, y_start:y_end] = 100.0

    # Precompute constant for Laplacian scaling (using dx for both axes)
    inv_dx2 = 1.0 / (dx * dx)

    for _ in range(steps):
        # Compute the 5‑point discrete Laplacian using slice operations.
        laplacian = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        ) * inv_dx2
        # Update the interior points; boundaries remain fixed (Dirichlet)
        u[1:-1, 1:-1] += alpha * laplacian * dt

    # Synchronise to ensure all GPU operations are complete
    cp.cuda.Stream.null.synchronize()
    return u


def main() -> None:
    """Entry point for script execution.

    Solves the heat equation and prints the resulting temperature field
    to stdout. If run from an interactive environment, consider visualising
    the result using matplotlib by converting the returned CuPy array to
    a NumPy array with ``.get()``.
    """
    final_field = heat_equation()
    # Transfer the result back to CPU memory for inspection
    host_field = cp.asnumpy(final_field)
    # Print a summary of the final temperature distribution
    print("Simulation complete. Sample values:")
    print(host_field)


if __name__ == "__main__":
    main()