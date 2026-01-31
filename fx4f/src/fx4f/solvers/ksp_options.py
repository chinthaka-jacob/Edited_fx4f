from petsc4py import PETSc
from treelog4dolfinx.logging_utils import log
import numpy as np

import dolfinx

from typing import Callable
from numpy.typing import NDArray

__all__ = [
    "KSPSetterRegistry",
    "set_ksp_options_direct",
    "set_ksp_options_gmres_boomeramg",
    "set_ksp_options_cg_jacobi",
    "set_ksp_options_cg_boomeramg",
    "ksp_monitor",
    "set_pc_hypre_boomeramg",
    "get_fieldsplit_is",
]


# Easy access through handles:
class KSPSetterRegistry:
    """
    Singleton registry for KSP option setter functions.

    This registry manages different KSP solver configurations that can be
    registered from various modules and accessed by name.
    """

    _setters: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, setter: Callable) -> None:
        """
        Register a KSP option setter function.

        Parameters
        ----------
        name : str
            Name of the setter (e.g., 'direct', 'iterative', 'stokes_schur')
        setter : Callable
            Function that configures KSP options. Should accept at minimum:
            ksp (PETSc.KSP), options_prefix (str), and **kwargs
        """
        cls._setters[name] = setter

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Retrieve a KSP option setter by name.

        Parameters
        ----------
        name : str
            Name of the setter

        Returns
        -------
        Callable
            The registered setter function

        Raises
        ------
        ValueError
            If setter is not registered
        """
        if name not in cls._setters:
            available = ", ".join(cls._setters.keys())
            raise ValueError(f"Unknown KSP setter '{name}'. Available: {available}")
        return cls._setters[name]

    @classmethod
    def available_setters(cls) -> list[str]:
        """Return list of available setter names."""
        return list(cls._setters.keys())


def set_ksp_options_direct(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for the Krylov solver (linear solver). Attempts to
    set to `mumps`, or else to `superlu_dist`, depending on availability.

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace, optional
        Solver context (unused for direct solver), by default None
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set nullspace (unused for direct solver), by default False
    log_iterations : bool, optional
        Whether to log solver iterations, by default True
    """
    # From https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_cahn-hilliard.html
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
    if nullspace:
        opts[f"{prefix}mat_mumps_icntl_24"] = 1
        opts[f"{prefix}mat_mumps_icntl_25"] = 0

    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp.setFromOptions()
    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_ksp_options_gmres_boomeramg(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for the Krylov solver (linear solver). Sets to
    `gmres` iterative solver with HYPRE BoomerAMG preconditioning.

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace, optional
        Solver context (unused for this solver), by default None
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set nullspace (unused for this solver), by default False
    log_iterations : bool, optional
        Whether to log solver iterations, by default True
    """
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "gmres"
    opts[f"{prefix}ksp_pc_side"] = "right"
    set_pc_hypre_boomeramg(opts, prefix)
    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp.setFromOptions()
    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_ksp_options_cg_jacobi(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for the Krylov solver (linear solver). Sets to
    `cg` (Conjugate Gradient) iterative solver with Jacobi preconditioning.

    Suitable for symmetric positive definite (SPD) systems.

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace, optional
        Solver context (unused for this solver), by default None
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set nullspace (unused for this solver), by default False
    log_iterations : bool, optional
        Whether to log solver iterations, by default True
    """
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "cg"
    opts[f"{prefix}pc_type"] = "jacobi"
    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp.setFromOptions()
    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_ksp_options_cg_boomeramg(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for the Krylov solver (linear solver). Sets to
    `cg` (Conjugate Gradient) iterative solver with HYPRE BoomerAMG preconditioning.

    Suitable for symmetric positive definite (SPD) systems. More robust than
    Jacobi preconditioning for ill-conditioned problems.

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace, optional
        Solver context (unused for this solver), by default None
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set nullspace (unused for this solver), by default False
    log_iterations : bool, optional
        Whether to log solver iterations, by default True
    """
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "cg"
    set_pc_hypre_boomeramg(opts, prefix)
    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp.setFromOptions()
    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_pc_hypre_boomeramg(opts: PETSc.Options, prefix: str) -> None:
    """
    Sets HYPRE BoomerAMG preconditioner options.

    Parameters
    ----------
    opts : PETSc.Options
        PETSc options database.
    prefix : str
        Options prefix for the solver.
    """
    # fmt: off
    opts[f"{prefix}pc_type"] = "hypre"  # Hypre (algebraic multigrid) preconditioning
    opts[f"{prefix}pc_hypre_type"] = "boomeramg"  # Choice of implementation; https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html
    opts[f"{prefix}pc_hypre_boomeramg_no_CF"] = None  # Whether to use CF-relaxation
    opts[f"{prefix}pc_hypre_boomeramg_coarsen_type"] = "HMIS"  # Falgout, HMIS, PMIS. Former is default, latter two are more aggressive and parallelize well.
    opts[f"{prefix}pc_hypre_boomeramg_interp_type"] = "ext+i"  # Should match physics. Many online references use 'ext+i'.
    # opts[f"{prefix}pc_hypre_boomeramg_truncfactor"] = 0.3 # Coarsening during interpolation, similar to strong_threshold.
    opts[f"{prefix}pc_hypre_boomeramg_strong_threshold"] = "0.85"  # !Impactful! Relative value the matrix entry must be over to be kept. See https://mooseframework.inl.gov/application_usage/hypre.html
    opts[f"{prefix}pc_hypre_boomeramg_P_max"] = 4  # Max elements per row for interpolation operator, but higher behaves less accurately. May be set to 2
    opts[f"{prefix}pc_hypre_boomeramg_agg_nl"] = 1  # Number of levels to apply aggressive coarsening to. Up to 4 tends to be Ok for 3D.
    opts[f"{prefix}pc_hypre_boomeramg_agg_num_paths"] = 2  # Number of pathways that must be present to keep a connection. Lower coarses more aggressively. Keep below 6.
    # fmt: on


# Register default KSP option setters
KSPSetterRegistry.register("default", set_ksp_options_direct)
KSPSetterRegistry.register("direct", set_ksp_options_direct)
KSPSetterRegistry.register("gmres_boomeramg", set_ksp_options_gmres_boomeramg)
KSPSetterRegistry.register("cg_jacobi", set_ksp_options_cg_jacobi)
KSPSetterRegistry.register("cg_boomeramg", set_ksp_options_cg_boomeramg)


def get_fieldsplit_is(
    A: PETSc.Mat = None,
    solver_context=None,
) -> tuple[PETSc.IS, PETSc.IS]:
    """
    Get PETSc Index Sets for velocity and pressure fields in a mixed space.

    Supports two approaches:
    1. Extract IS from nested matrix structure (if A is nested)
    2. Build IS from collapsed submaps of mixed function space (from solver_context)

    Parameters
    ----------
    A : PETSc.Mat, optional
        The system matrix. If nested, IS will be extracted from matrix structure.
    solver_context : SimpleNamespace, optional
        Solver context with attributes:
        - W : dolfinx.fem.FunctionSpace
            Mixed function space (required if A is not nested).
        - WV_map : NDArray, optional
            Velocity subspace dofmap. If None, computed from W.sub(0).collapse().
        - WQ_map : NDArray, optional
            Pressure subspace dofmap. If None, computed from W.sub(1).collapse().

    Returns
    -------
    tuple[PETSc.IS, PETSc.IS]
        Index sets for velocity (u) and pressure (p) fields.
    """
    # Method 1: Extract from nested matrix
    if A is not None and A.getType() == "nest":
        nested_IS = A.getNestISs()
        return nested_IS[0][0], nested_IS[0][1]

    # Method 2: Build from function space
    if solver_context is None:
        raise ValueError("solver_context is required when A is not nested")

    W = solver_context.W
    WV_map = getattr(solver_context, "WV_map", None)
    WQ_map = getattr(solver_context, "WQ_map", None)

    if WV_map is None or WQ_map is None:
        # This takes time, better to supply as argument if already computed
        _, WV_map = W.sub(0).collapse()  # Assuming sub(0) is velocity
        _, WQ_map = W.sub(1).collapse()  # Assuming sub(1) is pressure

    dofs_u = np.array(
        np.setdiff1d(
            W.dofmap.index_map.local_to_global(np.array(WV_map)),
            W.dofmap.index_map.ghosts,
        ),
        dtype=np.int32,
    )  # Getting the global velocity dofs that live on this core
    dofs_p = np.array(
        np.setdiff1d(
            W.dofmap.index_map.local_to_global(np.array(WQ_map)),
            W.dofmap.index_map.ghosts,
        ),
        dtype=np.int32,
    )  # Getting the global pressure dofs that live on this core
    is_u = PETSc.IS().createGeneral(dofs_u, comm=W.mesh.comm).sort()
    is_p = PETSc.IS().createGeneral(dofs_p, comm=W.mesh.comm).sort()

    return is_u, is_p


def _apply_user_options(
    opts: PETSc.Options, user_options: dict[str, str], prefix: str
) -> None:
    """Forwards all options in the specified dict to the `PETSc.Options` object"""
    for key, val in user_options.items():
        opts[f"{prefix}{key}"] = val


def ksp_monitor(ksp: PETSc.KSP, it: int, residual_norm: float):
    log(f"KSP iteration: {it}, residual norm: {residual_norm:.3e}", level="debug")
