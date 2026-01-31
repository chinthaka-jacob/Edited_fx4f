"""Convergence tests for error norms using analytical solutions.

Tests verify that interpolation errors of smooth analytical fields 
converge at the expected finite element rates when the mesh is refined.

Style aligns with existing reference-solution tests: use Taylor–Green (2D)
and Ethier–Steinman (3D) pressure fields interpolated onto Lagrange spaces,
then measure errors via `error_estimation.error_norms.error_norm`.
"""

import numpy as np
import pytest

from dolfinx import fem

from fx4f.reference_solutions.Taylor_Green import TaylorGreen2D
from fx4f.reference_solutions.Ethier_Steinman import EthierSteinman

from fx4f.error_estimation import error_norm


def _rate(err_coarse: float, err_fine: float, h_coarse: float, h_fine: float) -> float:
    """Compute observed convergence rate between two mesh levels."""
    return np.log(err_coarse / err_fine) / np.log(h_coarse / h_fine)


def test_error_norm_with_ufl_expression():
    """Test lines 52–56: UFL expression path and error field computation.

    Verifies that when u_ex is a UFL expression (not a callable),
    the isinstance check on line 52 triggers fem.Expression wrapping,
    and lines 55–56 (array subtraction and norm dispatch) execute correctly.
    """
    import ufl

    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=8, ny=8)
    V = fem.functionspace(tg.domain, ("Lagrange", 1))

    # Create a UFL expression tied to the domain (e.g., a coordinate-based expression)
    x = ufl.SpatialCoordinate(tg.domain)
    u_ex_ufl = 0.1 * x[0] * ufl.sin(x[1])  # Simple UFL expression

    # Interpolate a function onto V
    fn_approx = fem.Function(V)
    fn_approx.interpolate(tg.get_field("p", t=0.0))

    # Call error_norm with UFL expression as exact solution
    # This triggers line 52 (isinstance check), line 53 (fem.Expression),
    # and line 55–56 (interpolation, subtraction, and norm dispatch)
    err = error_norm(fn_approx, u_ex_ufl, norm="L2")

    # Error should be non-zero since the exact solution is different
    assert err > 0, f"Error should be positive, got {err}"


def test_interpolate_exact_solution_type_error():
    """Test that invalid type for u_ex raises TypeError.

    Verifies that _interpolate_exact_solution rejects unsupported types.
    """
    from fx4f.error_estimation.error_norms import _interpolate_exact_solution

    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=4, ny=4)
    V = fem.functionspace(tg.domain, ("Lagrange", 1))
    W = fem.functionspace(tg.domain, ("Lagrange", 2))

    # Try to interpolate with an invalid type (e.g., a string)
    with pytest.raises(TypeError, match="u_ex must be fem.Function"):
        _interpolate_exact_solution("invalid", W)


def test_error_norm_with_function_exact_solution():
    """Test error_norm with fem.Function as exact solution.

    Verifies that fem.Function branch in _interpolate_exact_solution works.
    """
    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=8, ny=8)
    V = fem.functionspace(tg.domain, ("Lagrange", 1))

    # Create two functions: one constant (approx) and one varying (exact)
    fn_approx = fem.Function(V)
    fn_approx.x.array[:] = 1.0  # Constant function

    fn_exact = fem.Function(V)
    fn_exact.interpolate(tg.get_field("p", t=0.0))  # Varying function

    # Call error_norm with fem.Function as exact solution
    err = error_norm(fn_approx, fn_exact, norm="L2")

    # Error should be positive since the functions differ
    assert err > 0, f"Error should be positive, got {err}"


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("norm,expected", [("L2", "p+1"), ("H10", "p"), ("H1", "p")])
def test_taylor_green_pressure_convergence(p, norm, expected):
    """Taylor–Green 2D pressure: error decreases with expected rate on refinement.

    Meshes: (8,8) -> (16,16). Time snapshot t=0.2.
    Expected rates: L2 ~ h^{p+1}, H10/H1 ~ h^{p}.
    """
    t = 0.2
    e1, h1 = _tg_pressure_error(nx=8, ny=8, p=p, norm=norm, t=t)
    e2, h2 = _tg_pressure_error(nx=16, ny=16, p=p, norm=norm, t=t)

    assert e2 < e1, "Error should decrease on mesh refinement"

    r_obs = _rate(e1, e2, h1, h2)
    r_exp = (p + 1) if expected == "p+1" else p

    # Allow modest tolerance due to quadrature/integration details
    assert abs(r_obs - r_exp) <= 0.5, f"Observed rate {r_obs:.2f} vs expected {r_exp}"


@pytest.mark.parametrize("p", [1])
@pytest.mark.parametrize("norm,expected", [("L2", "p+1"), ("H10", "p"), ("H1", "p")])
def test_ethier_steinman_pressure_convergence(p, norm, expected):
    """Ethier–Steinman 3D pressure: expected convergence on refinement.

    Meshes: (4,4,4) -> (8,8,8). Time snapshot t=0.1.
    Expected rates: L2 ~ h^{p+1}, H10/H1 ~ h^{p}.
    """
    t = 0.1
    e1, h1 = _es_pressure_error(nx=4, ny=4, nz=4, p=p, norm=norm, t=t)
    e2, h2 = _es_pressure_error(nx=8, ny=8, nz=8, p=p, norm=norm, t=t)

    assert e2 < e1, "Error should decrease on mesh refinement"

    r_obs = _rate(e1, e2, h1, h2)
    r_exp = (p + 1) if expected == "p+1" else p

    assert abs(r_obs - r_exp) <= 0.5, f"Observed rate {r_obs:.2f} vs expected {r_exp}"


def _tg_pressure_error(nx: int, ny: int, p: int, norm: str, t: float):
    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=nx, ny=ny)
    V = fem.functionspace(tg.domain, ("Lagrange", p))
    fn = fem.Function(V)
    # Interpolate approximate solution onto degree-p space
    fn.interpolate(tg.get_field("p", t=t))
    # Compute error vs exact analytical field
    err = error_norm(fn, tg.get_field("p", t=t), norm=norm)
    h = tg.L / nx
    return err, h


def _es_pressure_error(nx: int, ny: int, nz: int, p: int, norm: str, t: float):
    es = EthierSteinman(L=2.0)
    es.create_mesh(nx=nx, ny=ny, nz=nz)
    V = fem.functionspace(es.domain, ("Lagrange", p))
    fn = fem.Function(V)
    fn.interpolate(es.get_field("p", t=t))
    err = error_norm(fn, es.get_field("p", t=t), norm=norm)
    h = es.L / nx
    return err, h
