import ufl
from ufl import grad, inner, dx
from dolfinx import fem, default_real_type
from dolfinx.fem.petsc import LinearProblem

from typing import Callable

__all__ = ["L2_projection", "H1_projection"]


def L2_projection(
    V: fem.FunctionSpace,
    f: fem.Function | Callable,
    quadrature_degree: int | None = None,
    clamp: bool = False,
) -> fem.Function:
    """
    Perform an L2 projection of the expression/function `f` onto the space
    `V`.

    Parameters
    ----------
    V : fem.FunctionSpace
        Space on which to project
    f : fem.Function | Callable
        Expression/function to project
    quadrature_degree : int | None, optional
        Quadrature degree of integration carried out for projection, by
        default None
    clamp : bool, optional
        Whether to strongly impose the boundary dofs, by default False

    TODO: implement clamp

    Returns
    -------
    fem.Function
        Function in V that is the projection of `f`
    """
    bcs = []
    if clamp:
        raise NotImplementedError

    dX = (
        dx
        if quadrature_degree is None
        else ufl.Measure(
            "dx", V.mesh, metadata={"quadrature_degree": quadrature_degree}
        )
    )

    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dX

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="L2projection_",
    )
    return problem.solve()


def H1_projection(
    V: fem.FunctionSpace,
    f: fem.Function | Callable,
    quadrature_degree: int | None = None,
    clamp: bool = False,
    L2_weight: float = 1,
) -> fem.Function:
    """
    Perform an H1 projection of the expression/function `f` onto space V.

    Minimizes ||u - f||_H1 over u ∈ V, where the H1 norm is weighted:
    ||·||_H1² = w * ||·||_L2² + ||∇·||_L2².

    Parameters
    ----------
    V : fem.FunctionSpace
        Space on which to project.
    f : fem.Function | Callable
        Expression or function to project.
    quadrature_degree : int | None, optional
        Quadrature degree for integration, by default None (auto).
    clamp : bool, optional
        Whether to strongly impose boundary dofs, by default False.
        Not implemented.
    L2_weight : float, optional
        Weight w in the H1 norm, by default 1.

    Returns
    -------
    fem.Function
        Projection of f onto V.

    Raises
    ------
    NotImplementedError
        If clamp is True.
    """
    bcs = []
    if clamp:
        raise NotImplementedError

    dX = (
        dx
        if quadrature_degree is None
        else ufl.Measure(
            "dx", V.mesh, metadata={"quadrature_degree": quadrature_degree}
        )
    )

    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    w = fem.Constant(V.mesh, default_real_type(L2_weight))
    a = inner(w * u, v) * dx + inner(grad(u), grad(v)) * dx
    L = inner(w * f, v) * dx + inner(grad(f), grad(v)) * dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="H1_projection_",
    )
    return problem.solve()


def H10_projection(
    V: fem.FunctionSpace,
    f: fem.Function | Callable,
    quadrature_degree: int | None = None,
) -> fem.Function:
    """
    Perform an H¹₀ semi-norm projection of f onto space V.

    Minimizes ||∇u - ∇f||_L2 over u ∈ V. Requires boundary conditions.

    Parameters
    ----------
    V : fem.FunctionSpace
        Space on which to project.
    f : fem.Function | Callable
        Expression or function to project.
    quadrature_degree : int | None, optional
        Quadrature degree for integration, by default None (auto).

    Returns
    -------
    fem.Function
        Projection of f onto V.

    Raises
    ------
    NotImplementedError
        Always raised; requires boundary condition specification.
    """
    bcs = []
    raise NotImplementedError  # Must define BC

    dX = (
        dx
        if quadrature_degree is None
        else ufl.Measure(
            "dx", V.mesh, metadata={"quadrature_degree": quadrature_degree}
        )
    )

    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = inner(grad(f), grad(v)) * dX

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="H10projection_",
    )
    return problem.solve()
