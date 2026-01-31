from dolfinx import fem
from ufl import sym, dx, ds, nabla_grad, grad, inner, Measure, FacetNormal, as_vector
from mpi4py import MPI
from functools import lru_cache

__all__ = ["kinetic_energy", "dissipation_rate", "aerodynamic_forces_2D"]


def kinetic_energy(
    u: fem.Function, rho: float | fem.Constant | fem.Function = 1
) -> float:
    """
    Kinetic energy associated to the velocity field `u`, per the domain
    integral of $1/2 rho u^2$.

    Parameters
    ----------
    u : fem.Function
        Velocity field
    rho : float | fem.Constant, optional | fem.Function
        Mass density, by default 1

    Returns
    -------
    float
        Total kinetic energy
    """
    comm = u.function_space.mesh.comm
    E_kin = fem.assemble_scalar(_E_kin_form(rho, u))
    return comm.allreduce(E_kin, op=MPI.SUM)


@lru_cache
def _E_kin_form(rho, u):
    """Cached kinetic energy form: 0.5 * rho * u·u.

    Avoids recompilation of the form on repeated calls with the same rho.
    """
    if isinstance(rho, (int, float)):
        rho = fem.Constant(u.function_space.mesh, float(rho))
    return fem.form(0.5 * rho * inner(u, u) * dx)


def dissipation_rate(u: fem.Function, mu: float | fem.Constant | fem.Function) -> float:
    """
    Dissipation rate associated to the velocity field `u`, per the domain
    integral of $2 nu E(u):E(u)$.

    Parameters
    ----------
    u : fem.Function
        Velocity field
    mu : float | fem.Constant | fem.Function
        Dynamic viscosity

    Returns
    -------
    float
        Dissipation rate
    """
    comm = u.function_space.mesh.comm
    D = fem.assemble_scalar(_D_form(mu, u))
    return comm.allreduce(D, op=MPI.SUM)


@lru_cache
def _D_form(mu, u):
    """Cached dissipation form: 2 * mu * E(u):E(u) with E = sym(∇u).

    Avoids recompilation of the form on repeated calls with the same mu.
    """
    if isinstance(mu, (int, float)):
        mu = fem.Constant(u.function_space.mesh, float(mu))
    return fem.form(2 * mu * inner(sym(nabla_grad(u)), sym(nabla_grad(u))) * dx)


def aerodynamic_forces_2D(
    u: fem.Function,
    p: fem.Function,
    ds: Measure,
    mu: float | fem.Constant | fem.Function,
) -> tuple[float, float]:
    """
    Get the loads due acting on surface `ds` due to Newtonian stress.

    NOTE: Expression for forces from https://doi.org/10.1007/978-3-322-89849-4_39.
    I am surprised this works so well, and appears to work better than a
    more conventional (stress*n integral) approach.

    Parameters
    ----------
    u : fem.Function
        Velocity field
    p : fem.Function
        Pressure field
    ds : Measure
        Area of integration
    mu : float | fem.Constant | fem.Function
        Dynamic viscosity

    Returns
    -------
    tuple[float,float]
        (Fx,Fy)
    """
    comm = u.function_space.mesh.comm
    Fx_form, Fy_form = _F_forms_2D(mu, u, p, ds)
    Fx, Fy = [fem.assemble_scalar(F_form) for F_form in [Fx_form, Fy_form]]
    return comm.allreduce(Fx, op=MPI.SUM), comm.allreduce(Fy, op=MPI.SUM)


@lru_cache
def _F_forms_2D(mu, u, p, ds):
    """Cached 2D aerodynamic force forms (Fx, Fy).

    Avoids recompilation on repeated calls with the same parameters.
    """
    if isinstance(mu, (int, float)):
        mu = fem.Constant(u.function_space.mesh, float(mu))
    domain = u.function_space.mesh
    n = FacetNormal(domain)
    n_out = -n
    u_t = inner(as_vector((n_out[1], -n_out[0])), u)
    Fx = fem.form((mu * inner(grad(u_t), n_out) * n_out[1] - p * n_out[0]) * ds)
    Fy = fem.form((-mu * inner(grad(u_t), n_out) * n_out[0] - p * n_out[1]) * ds)
    return (Fx, Fy)
