from dolfinx import fem, default_real_type
import ufl
from mpi4py import MPI
from functools import lru_cache

from treelog4dolfinx import log
from fx4f.miscellaneous.mesh_operations import get_domain_volume

__all__ = ["zero_mean_pressure"]


def zero_mean_pressure(p_sol: fem.Function) -> None:
    """
    Post-process the input pressure (in place) to have zero mean over its domain.

    Integration forms and domain volume are cached automatically using @lru_cache
    to avoid recompilation on repeated calls with the same pressure function or mesh.

    Parameters
    ----------
    p_sol : fem.Function
        Pressure solution function to be normalized to have zero mean.
    """
    domain = p_sol.function_space.mesh
    vol = get_domain_volume(domain)
    p_form = _get_pressure_form(p_sol)
    p_mean = domain.comm.allreduce(fem.assemble_scalar(p_form), op=MPI.SUM) / vol
    p_sol.x.array[:] -= p_mean


@lru_cache(maxsize=128)
def _get_pressure_form(p_sol: fem.Function) -> fem.Form:
    """
    Cache the pressure integration form for a given pressure function.

    Parameters
    ----------
    p_sol : fem.Function
        Pressure solution function.

    Returns
    -------
    object
        Pressure integration form (dolfinx.fem.Form).
    """
    return fem.form(p_sol * ufl.dx)
