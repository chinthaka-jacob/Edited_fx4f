"""FEniCSx for Flow (fx4f)

A collection of reusable CFD components for FEniCSx, including:
- Reference solutions and analytical benchmarks
- IO utilities (checkpointing, plotting)
- Solver configurations (IMEX Runge-Kutta, KSP/SNES options)
- Error estimation and convergence analysis
- Fluid mechanics utilities (measures, initial conditions, postprocessing)
- Turbulence statistics and processing
- Mesh operations and projection operators

Designed to be used as a git submodule in simulation projects.
"""

__version__ = "0.10.0"

__all__ = [
    "error_estimation",
    "fluid_mechanics",
    "io",
    "miscellaneous",
    "reference_solutions",
    "solvers",
    "turbulence",
]
