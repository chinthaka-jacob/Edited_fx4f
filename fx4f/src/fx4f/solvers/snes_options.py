from .ksp_options import _apply_user_options

from petsc4py import PETSc
from treelog4dolfinx.logging_utils import log

from typing import Callable

__all__ = [
    "SNESSetterRegistry",
    "set_snes_options_newtonls",
    "snes_monitor",
]


# Singleton registry for SNES option setters
class SNESSetterRegistry:
    """
    Singleton registry for SNES option setter functions.

    This registry manages different SNES solver configurations that can be
    registered from various modules and accessed by name.
    """

    _setters: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, setter: Callable) -> None:
        """
        Register a SNES option setter function.

        Parameters
        ----------
        name : str
            Name of the setter (e.g., 'default', 'newtonls')
        setter : Callable
            Function that configures SNES options. Should accept at minimum:
            snes (PETSc.SNES), options (dict), log_iterations (bool), options_prefix (str), and **kwargs
        """
        cls._setters[name] = setter

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Retrieve a SNES option setter by name.

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
            raise ValueError(f"Unknown SNES setter '{name}'. Available: {available}")
        return cls._setters[name]

    @classmethod
    def available_setters(cls) -> list[str]:
        """Return list of available setter names."""
        return list(cls._setters.keys())


def set_snes_options_newtonls(
    snes: PETSc.SNES,
    options: dict[str, str] | None = None,
    log_iterations: bool = True,
    options_prefix: str = "",
    **kwargs,
) -> None:
    """
    Option setter for the SNES solver. Newton solver with line search

    Parameters
    ----------
    snes : PETSc.SNES
        Scalable Nonlinear Equation Solver object.
    options : dict[str, str], optional
        Additional user options, by default {}
    log_iterations : bool, optional
        Whether to log iteration counts, by default True
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    """
    # From https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_cahn-hilliard.html
    opts = PETSc.Options()
    prefix = snes.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    snes.setOptionsPrefix(prefix)

    opts[f"{prefix}snes_type"] = "newtonls"
    opts[f"{prefix}snes_linesearch_type"] = "none"
    opts[f"{prefix}snes_atol"] = 0
    opts[f"{prefix}snes_rtol"] = 1e-6
    opts[f"{prefix}snes_error_if_not_converged"] = True
    _apply_user_options(opts, options or {}, prefix)

    # Set options
    snes.setFromOptions()
    if log_iterations:
        snes.setMonitor(snes_monitor)


# Register default SNES option setters
SNESSetterRegistry.register("default", set_snes_options_newtonls)
SNESSetterRegistry.register("newtonls", set_snes_options_newtonls)


def snes_monitor(snes: PETSc.SNES, it: int, residual_norm: float):
    log(f"SNES iteration: {it}, residual norm: {residual_norm:.3e}", level="debug")
