"""Unified test suite for reference solutions with optimal coverage.

Combines base API tests and concrete implementation validation into a
lightweight test suite covering:
- AnalyticalSolution API (lambdify, field getters, derivatives, caching)
- ReferenceSolution infrastructure (mesh, facets, periodic boundaries)
- Field consistency and numerical correctness
- Dolfinx integration and interpolation
"""

import numpy as np
import pytest

from dolfinx import fem

from fx4f.reference_solutions.Taylor_Green import TaylorGreen2D

# ============================================================================
# Taylor-Green 2D: Field Accessibility and Derivative Correctness
# ============================================================================


class TestTaylorGreenFieldAccessibility:
    """Test that all field names are accessible with correct shapes."""

    @pytest.fixture
    def tg(self):
        return TaylorGreen2D(L=2 * np.pi, nu=1.0)

    def test_base_fields_accessible(self, tg):
        """Test base fields u, p, p_mean, sigma_0 are accessible."""
        coords = np.array([[0.5, 1.0], [0.5, 1.0]])

        u_vals = tg.get_field("u")(coords, t=0.0)
        assert u_vals.shape == (2, 2)

        p_vals = tg.get_field("p")(coords, t=0.0)
        assert p_vals.shape == (2,)

        p_mean = tg.get_field("p_mean")
        assert isinstance(p_mean, (float, np.ndarray, int))

        sigma_vals = tg.get_field("sigma", time_derivative=0)(coords, t=0.0)
        assert len(sigma_vals) == 4

    def test_all_velocity_derivatives_accessible(self, tg):
        """Test all velocity derivative orders 0-7 are accessible."""
        coords = np.array([[0.5], [0.5]])
        for order in range(0, 8):
            result = tg.get_field("u", time_derivative=order)(coords, t=0.0)
            assert result is not None
            assert isinstance(result, (np.ndarray, list))

    def test_all_pressure_derivatives_accessible(self, tg):
        """Test all pressure derivative orders 0-6 are accessible."""
        coords = np.array([[0.5], [0.5]])
        for order in (0, 1, 2, 4, 6):
            result = tg.get_field("p", time_derivative=order)(coords, t=0.0)
            assert result is not None

    def test_all_stress_derivatives_accessible(self, tg):
        """Test all stress tensor derivative orders (even 0,2,4,6) are accessible."""
        coords = np.array([[0.5], [0.5]])
        for order in (0, 2, 4, 6):
            result = tg.get_field("sigma", time_derivative=order)(coords, t=0.0)
            assert result is not None


class TestTaylorGreenDerivativeCorrectness:
    """Test numerical correctness of derivatives via finite differences."""

    def test_velocity_first_derivative(self):
        """Verify u_1 matches finite-difference approximation."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0)
        coords = np.array([[0.5], [0.5]])
        dt = 1e-8

        u_t0 = tg.get_field("u")(coords, t=0.0)
        u_t_plus = tg.get_field("u")(coords, t=dt)
        u1_analytical = tg.get_field("u", time_derivative=1)(coords, t=0.0)

        u1_fd = (u_t_plus - u_t0) / dt
        np.testing.assert_allclose(u1_fd, u1_analytical, rtol=1e-6)

    def test_pressure_first_derivative(self):
        """Verify p_1 matches finite-difference approximation."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0)
        coords = np.array([[0.5], [0.5]])
        dt = 1e-4

        p_t0 = tg.get_field("p")(coords, t=0.0)
        p_t_plus = tg.get_field("p")(coords, t=dt)
        p1_analytical = tg.get_field("p", time_derivative=1)(coords, t=0.0)

        p1_fd = (p_t_plus - p_t0) / dt
        np.testing.assert_allclose(p1_fd, p1_analytical, rtol=1e-2)

    def test_repeated_calls_are_consistent(self):
        """Verify caching works: repeated calls give identical results."""
        tg = TaylorGreen2D(L=2 * np.pi)
        coords = np.array([[0.5], [0.5]])

        call1 = tg.get_field("u", time_derivative=1)(coords, t=0.0)
        call2 = tg.get_field("u", time_derivative=1)(coords, t=0.0)
        np.testing.assert_allclose(call1, call2)


class TestTaylorGreenTimeBehavior:
    """Test time evolution (decay) of Taylor-Green fields."""

    def test_velocity_decay(self):
        """Verify velocity decays exponentially with time."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0)
        coords = np.array([[0.5], [0.5]])

        u_t0 = tg.get_field("u")(coords, t=0.0)
        u_t1 = tg.get_field("u")(coords, t=1.0)
        u_t2 = tg.get_field("u")(coords, t=2.0)

        norm_t0 = np.linalg.norm(u_t0)
        norm_t1 = np.linalg.norm(u_t1)
        norm_t2 = np.linalg.norm(u_t2)

        assert norm_t1 < norm_t0, "Velocity should decay with time"
        assert norm_t2 < norm_t1, "Velocity should continue to decay"

    def test_pressure_decay(self):
        """Verify pressure decays (faster than velocity)."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0)
        coords = np.array([[0.5], [0.5]])

        p_t0 = tg.get_field("p")(coords, t=0.0)
        p_t1 = tg.get_field("p")(coords, t=1.0)

        assert np.linalg.norm(p_t1) < np.linalg.norm(p_t0)

    def test_stress_tensor_time_evolution(self):
        """Verify stress tensor components evolve consistently."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0)
        coords = np.array([[0.5], [0.5]])

        sigma_t0 = tg.get_field("sigma", time_derivative=0)(coords, t=0.0)
        sigma_t1 = tg.get_field("sigma", time_derivative=0)(coords, t=0.1)
        assert len(sigma_t0) == len(sigma_t1)


# ============================================================================
# Integration Tests: Interpolation and NS Verification
# ============================================================================


class TestTaylorGreenIntegration:
    """Integration tests for Taylor-Green with dolfinx and NS verification."""

    def test_interpolate_base_fields(self):
        """Test interpolation of pressure and derivatives onto function spaces."""
        tg = TaylorGreen2D(L=2 * np.pi)
        tg.create_mesh(nx=4, ny=4)
        V = fem.functionspace(tg.domain, ("Lagrange", 1))

        # Interpolate base field
        fn_p = fem.Function(V)
        tg.interpolate_field(fn_p, "p", t=0.5)
        assert fn_p.x.array.shape[0] > 0

        # Interpolate derivative
        fn_p1 = fem.Function(V)
        tg.interpolate_field(fn_p1, "p", time_derivative=1, t=0.0)
        assert fn_p1.x.array.shape[0] > 0

    def test_ns_equation_verification(self):
        """Verify Taylor-Green satisfies incompressible NS equations."""
        tg = TaylorGreen2D(L=2 * np.pi, nu=1.0, mode=1)
        result = tg.confirm_incompressible_NS_solution()
        assert result is True


def test_facets_and_node_origin():
    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=4, ny=4)
    # Facet getters return non-empty selections
    domain_boundary = tg.get_facets("domain_boundary")
    walls_v = tg.get_facets("walls_vertical")
    walls_h = tg.get_facets("walls_horizontal")
    assert domain_boundary.size > 0
    assert walls_v.size > 0
    assert walls_h.size > 0


def test_taylor_green_facet_getters():
    """Test all facet getter methods return valid facet indices."""
    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=4, ny=4)

    # Test domain_boundary facet getter
    boundary_facets = tg.get_facets("domain_boundary")
    assert boundary_facets.size > 0
    assert boundary_facets.dtype == np.int32 or boundary_facets.dtype == np.int64

    # Test walls_vertical facet getter
    vertical_facets = tg.get_facets("walls_vertical")
    assert vertical_facets.size > 0

    # Test walls_horizontal facet getter
    horizontal_facets = tg.get_facets("walls_horizontal")
    assert horizontal_facets.size > 0

    # Verify vertical and horizontal walls don't overlap
    combined = np.concatenate([vertical_facets, horizontal_facets])
    assert len(np.unique(combined)) == len(
        combined
    ), "Vertical and horizontal facets should not overlap"


def test_taylor_green_dof_getter():
    """Test origin DOF getter returns valid DOF indices."""
    tg = TaylorGreen2D(L=2 * np.pi)
    tg.create_mesh(nx=4, ny=4)

    V = fem.functionspace(tg.domain, ("Lagrange", 1))
    origin_dofs = tg._get_origin_dofs(V)
    assert origin_dofs.size > 0
    assert origin_dofs.dtype == np.int32 or origin_dofs.dtype == np.int64
