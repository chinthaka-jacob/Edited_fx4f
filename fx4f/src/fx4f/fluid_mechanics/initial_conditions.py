from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem
from ufl import (
    TestFunctions,
    TrialFunctions,
    Measure,
    dot,
    sym,
    dx,
    div,
    nabla_grad,
    inner,
    FacetNormal,
)
from math import comb
from fx4f.miscellaneous.mesh_operations import get_element_sizes

__all__ = ["compute_dudt_p_from_u"]

"""Initial condition helpers for incompressible Navier-Stokes.

Computes time derivatives (dudt, dduddt, ...) and pressure from a velocity
field via weak formulation of the NS equations.
"""


def compute_dudt_p_from_u(
    W: fem.FunctionSpace,
    u: fem.Function | list[fem.Function],
    mu: float,
    bcs: list[fem.DirichletBC] = [],
    ds_impermeable: Measure | None = None,
    F: fem.Form | None = None,
    rho: float = 1,
    stabilization: bool = False,
) -> fem.Function:
    """
    For the incompressible Navier-Stokes equations, given a velocity field,
    compute the associated acceleration and pressure field.

    May also be used to compute dduddt and dpdt, or higher, through
    specifying the correct bcs, F, and providing a list of [u, du/dt, ...]
    in place of u.

    Parameters
    ----------
    W : fem.FunctionSpace
        mixed function space VxQ
    u : fem.Function | list[fem.Function]
        velocity function, or list [u,dudt,dduddt,...] if n-th order
        time-derivative must be computed
    mu : float
        viscosity (by default rho=1, so mu=nu)
    bcs : list[fem.DirichletBC] | None, optional
        Boundary conditions on dudt, by default None (no conditions).
    ds_impermeable : Measure | None, optional
        Surface measure where impermeability conditions are enforced with
        Nitsche's method, by default None
    F : fem.Form | None, optional
        Right-hand-side of the incompressible Navier-Stokes equations, by
        default None
    rho : float, optional
        Density, by default 1
    stabilization : bool, optional
        Whether to add LSIC stabilization, by default False

    Returns
    -------
    fem.Function
        Mixed function (dudt,p) in W.
    """
    # Argument parsing to enable higher-order time derivatives later on.
    us = u if isinstance(u, list) else [u]
    u = u[-1] if isinstance(u, list) else u

    # Define trial and test functions
    dudt, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Operators
    def epsilon(u):
        return sym(nabla_grad(u))

    # LHS:
    a = dot(rho * dudt, v) * dx - dot(div(v), p) * dx - dot(div(dudt), q) * dx
    if stabilization:
        h = get_element_sizes(W.mesh)
        a += 10 * h**2 * rho * inner(div(dudt), div(v)) * dx
        # a += 1E-3*inner( 2*mu*epsilon(dudt), epsilon(v) )*dx

    if ds_impermeable is not None:
        # Nitsche impermeability condition
        h = get_element_sizes(W.mesh)
        n = FacetNormal(W.mesh)
        a += (
            inner(p, dot(v, n)) * ds_impermeable
            + inner(q, dot(dudt, n)) * ds_impermeable
            + 10 * h * inner(dot(dudt, n), dot(v, n)) * ds_impermeable
        )

    # RHS
    L = 1e-16 * q * dx
    L = -inner(2 * mu * epsilon(u), epsilon(v)) * dx
    if F is not None:
        L += F
    for j in range(0, len(us)):
        L -= comb(len(us), j) * inner(rho * dot(us[-j - 1], nabla_grad(us[j])), v) * dx

    # Solve
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="ICns_",
    )
    return problem.solve()
