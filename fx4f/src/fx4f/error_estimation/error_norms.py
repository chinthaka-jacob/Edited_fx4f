import dolfinx.fem as fem
from basix.ufl import element
import ufl
import numpy as np
from mpi4py import MPI
from functools import lru_cache

from typing import Callable, Union

__all__ = ["error_norm", "norm_L2", "norm_H10", "norm_H1"]


def error_norm(
    uh: fem.Function,
    u_ex: Union[fem.Function, Callable, ufl.core.expr.Expr],
    norm: str = "L2",
    degree_raise: int = 3,
) -> float:
    """
    Compute the normed difference between two functions via higher-order projection.

    Interpolates both `uh` and `u_ex` onto a higher-order function space before
    computing the norm of their difference. This ensures accurate integration for
    convergence studies.

    Parameters
    ----------
    uh : fem.Function
        Discrete approximation.
    u_ex : fem.Function | Callable | ufl.core.expr.Expr
        Exact solution. Can be a fem.Function, Python callable, or UFL expression.
    norm : str, optional
        Type of norm to compute: "L2", "H10", or "H1", by default "L2"
    degree_raise : int, optional
        Polynomial degree increase of the projection space, by default 3.
        Larger values improve accuracy at computational cost.

    Returns
    -------
    float
        Error norm value (non-negative).

    Raises
    ------
    KeyError
        If `norm` is not one of "L2", "H10", or "H1".
    TypeError
        If `u_ex` is not fem.Function, Callable, or ufl.core.expr.Expr.

    Examples
    --------
    >>> err_L2 = error_norm(uh_approx, u_ex_exact, norm="L2")
    >>> err_H1 = error_norm(uh_approx, u_ex_exact, norm="H1")
    """
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    shape = uh.ufl_shape
    domain = uh.function_space.mesh
    We = element(family, domain.basix_cell(), degree + degree_raise, shape=shape)
    W = fem.functionspace(domain, We)

    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution with type-checked helper
    errorfield = _interpolate_exact_solution(u_ex, W)

    # Compute the error in the higher order function space
    errorfield.x.array[:] -= u_W.x.array
    norms = {"L2": norm_L2, "H10": norm_H10, "H1": norm_H1}
    return norms[norm](errorfield)


def norm_L2(field: fem.Function) -> float:
    """
    Compute L2 norm of a field: ||field||_L2 = sqrt(∫ field² dx).

    Parameters
    ----------
    field : fem.Function
        Function of which to compute the norm.

    Returns
    -------
    float
        L2 norm value (non-negative).

    Notes
    -----
    The form is cached per field to avoid recompilation on repeated calls.
    Uses MPI allreduce for distributed parallel assembly.
    """
    domain = field.function_space.mesh
    error = _form_L2(field)
    error_local = fem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


@lru_cache
def _form_L2(field: fem.Function) -> fem.Form:
    """Cache L2 norm bilinear form: ∫ field² dx.

    This cached form avoids recompilation when computing L2 norms
    on the same field multiple times.
    """
    return fem.form(ufl.inner(field, field) * ufl.dx)


def norm_H10(field: fem.Function) -> float:
    """
    Compute H¹₀ semi-norm of a field: |field|_H10 = sqrt(∫ ∇field·∇field dx).

    The H¹₀ semi-norm measures only the gradient (derivative) contribution.

    Parameters
    ----------
    field : fem.Function
        Function of which to compute the norm.

    Returns
    -------
    float
        H¹₀ semi-norm value (non-negative).

    Notes
    -----
    The form is cached per field to avoid recompilation on repeated calls.
    Uses MPI allreduce for distributed parallel assembly.
    """
    domain = field.function_space.mesh
    error = _form_H10(field)
    error_local = fem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


@lru_cache
def _form_H10(field: fem.Function) -> fem.Form:
    """Cache H¹₀ semi-norm bilinear form: ∫ ∇field·∇field dx.

    This cached form avoids recompilation when computing H¹₀ semi-norms
    on the same field multiple times.
    """
    return fem.form(ufl.inner(ufl.grad(field), ufl.grad(field)) * ufl.dx)


def norm_H1(field: fem.Function) -> float:
    """
    Compute H¹ norm of a field: ||field||_H1 = sqrt(∫ field² + ∇field·∇field dx).

    The H¹ norm (squared) equals the sum of the L² norm and H¹₀ semi-norm (squared).

    Parameters
    ----------
    field : fem.Function
        Function of which to compute the norm.

    Returns
    -------
    float
        H¹ norm value (non-negative).

    Notes
    -----
    The form is cached per field to avoid recompilation on repeated calls.
    Uses MPI allreduce for distributed parallel assembly.
    H¹ norm >= H¹₀ semi-norm >= 0.
    """
    domain = field.function_space.mesh
    error = _form_H1(field)
    error_local = fem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


@lru_cache
def _form_H1(field: fem.Function) -> fem.Form:
    """Cache H¹ norm bilinear form: ∫ field² + ∇field·∇field dx.

    This cached form avoids recompilation when computing H¹ norms
    on the same field multiple times.
    """
    return fem.form(
        (ufl.inner(field, field) + ufl.inner(ufl.grad(field), ufl.grad(field))) * ufl.dx
    )


def _interpolate_exact_solution(
    u_ex: Union[fem.Function, Callable, ufl.core.expr.Expr],
    W: fem.FunctionSpace,
) -> fem.Function:
    """Interpolate exact solution onto higher-order function space.

    Handles three input types with appropriate interpolation methods:
    - fem.Function: direct interpolation via fem.Function.interpolate()
    - ufl.core.expr.Expr: wrapped in fem.Expression for point evaluation
    - Callable: direct interpolation via fem.Function.interpolate()

    Parameters
    ----------
    u_ex : fem.Function | Callable | ufl.core.expr.Expr
        Exact solution in any supported form.
    W : fem.FunctionSpace
        Target function space (higher-order, used for error projection).

    Returns
    -------
    fem.Function
        Interpolated exact solution in space W.

    Raises
    ------
    TypeError
        If u_ex is not fem.Function, Callable, or ufl.core.expr.Expr.

    Notes
    -----
    fem.Function is checked first since it is also an instance of
    ufl.core.expr.Expr in DOLFINx's type hierarchy.
    """
    u_ex_W = fem.Function(W)

    if isinstance(u_ex, fem.Function):
        # fem.Function: direct interpolation (check before UFL since Function is also Expr)
        u_ex_W.interpolate(u_ex)
    elif isinstance(u_ex, ufl.core.expr.Expr):
        # UFL expression: wrap in fem.Expression for evaluation
        u_expr = fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    elif callable(u_ex):
        # Python callable: direct interpolation
        u_ex_W.interpolate(u_ex)
    else:
        raise TypeError(
            f"u_ex must be fem.Function, Callable, or ufl.core.expr.Expr, "
            f"got {type(u_ex).__name__}"
        )

    return u_ex_W
