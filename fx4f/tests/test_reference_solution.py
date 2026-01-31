"""Test suite for reference solutions.

- AnalyticalSolution API (lambdify, field getters, derivatives, caching)
- ReferenceSolution infrastructure (mesh, facets, periodic boundaries)
- Field consistency and numerical correctness
- Dolfinx integration and interpolation
"""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh

from fx4f.reference_solutions.reference_solution import (
    AnalyticalIncompressibleNavierStokesSolution,
    ReferenceSolution,
    _stack,
    x,
    y,
    z,
    t,
)
from fx4f.reference_solutions.Taylor_Green import TaylorGreen2D


# ============================================================================
# Minimal Stubs for Base API Testing
# ============================================================================


class _AnalyticalStub2D(AnalyticalIncompressibleNavierStokesSolution):
    """Minimal 2D analytical solution with simple polynomials."""

    def __init__(self):
        super().__init__(dim=2)
        self.rho = 1.0
        self.nu = 1.0
        self.mu = 1.0
        self.u = [x + t, y]
        self.p = x - y + t
        self.p_mean = 0
        self._register_NS_fields()


class _AnalyticalStub3D(AnalyticalIncompressibleNavierStokesSolution):
    """Minimal 3D analytical solution mirroring the 2D stub."""

    def __init__(self):
        super().__init__(dim=3)
        self.rho = 1.0
        self.nu = 1.0
        self.mu = 1.0
        self.u = [x, y + t, z]
        self.p = x + y + z + t
        self.p_mean = 0
        self._register_NS_fields()


class _ReferenceStub(ReferenceSolution):
    """ReferenceSolution with minimal periodic indicators and facet getters."""

    def __init__(self, domain: mesh.Mesh):
        super().__init__()
        self.domain = domain
        domain.topology.create_entities(domain.topology.dim - 1)
        self._facets_getters = {
            "all": lambda: np.arange(
                domain.topology.index_map(domain.topology.dim - 1).size_local
            )
        }
        self._elements_getters = {}
        self._dofs_getters = {}

    def node_origin(self, x_arr: np.ndarray) -> np.ndarray:
        return np.isclose(x_arr[0], 0.0) & np.isclose(x_arr[1], 0.0)

    def periodic_boundary_indicator(self, x_arr: np.ndarray) -> np.ndarray:
        return np.isclose(x_arr[0], 0.0)

    def periodic_relation(self, x_arr: np.ndarray) -> np.ndarray:
        out = np.copy(x_arr)
        out[0] = 1.0
        return out


# ============================================================================
# Base API Tests (lambdify, field getters, stacking)
# ============================================================================


class TestAnalyticalSolutionAPI:
    """Test AnalyticalSolution API: lambdify, field access, shapes."""

    def test_2d_field_evaluation(self):
        """Test 2D solution field evaluation and shape consistency."""
        sol = _AnalyticalStub2D()
        coords = np.array([[0.0, 1.0], [1.0, -1.0]])

        # Base fields
        u_vals = sol.get_field("u")(coords, t=2.0)
        assert u_vals.shape == (2, 2)
        np.testing.assert_allclose(u_vals[0], [2.0, 3.0])
        np.testing.assert_allclose(u_vals[1], [1.0, -1.0])

        p_vals = sol.get_field("p")(coords, t=2.0)
        np.testing.assert_allclose(p_vals, [1.0, 4.0])

        # Derivatives
        u1_vals = sol.get_field("u", time_derivative=1)(coords, t=2.0)
        np.testing.assert_allclose(u1_vals, [1.0, 0.0])

    def test_3d_field_evaluation(self):
        """Test 3D solution field evaluation."""
        sol = _AnalyticalStub3D()
        coords = np.array([[0.0, 1.0], [1.0, -1.0], [0.5, -0.5]])

        u_vals = sol.get_field("u")(coords, t=1.5)
        assert u_vals.shape == (3, 2)
        np.testing.assert_allclose(u_vals[1], [2.5, 0.5])

        p_vals = sol.get_field("p")(coords, t=1.5)
        np.testing.assert_allclose(p_vals, [3.0, 1.0])

    def test_tensor_field_shape(self):
        """Test stress tensor returns correct flattened shape."""
        sol = _AnalyticalStub2D()
        coords = np.array([[0.5], [0.5]])
        sigma_vals = sol.get_field("sigma", time_derivative=0)(coords, t=0.0)
        assert len(sigma_vals) == sol.dim * sol.dim

    def test_stack_utility_function(self):
        """Test _stack handles mixed scalar/vector edge cases."""
        vec = np.array([1.0, 2.0])

        # Normal stacking
        out = _stack((vec, vec * 2))
        assert np.allclose(out, np.stack((vec, vec * 2)))

        # Mixed scalar/vector
        out_scalar = _stack((1.0, vec))
        assert isinstance(out_scalar, list)
        np.testing.assert_allclose(out_scalar[0], np.ones_like(vec))
        np.testing.assert_allclose(out_scalar[1], vec)

        # Near-zero treated as zero
        out_near_zero = _stack((1e-16, vec))
        np.testing.assert_allclose(out_near_zero[0], np.zeros_like(vec))

    def test_unknown_field_raises_keyerror(self):
        """Test that accessing nonexistent field raises KeyError."""
        sol = _AnalyticalStub2D()
        with pytest.raises(KeyError):
            sol.get_field("not_a_field")


# ============================================================================
# ReferenceSolution Infrastructure Tests (mesh, facets, periodicity)
# ============================================================================


class TestReferenceSolutionInfrastructure:
    """Test ReferenceSolution mesh and boundary handling."""

    def test_mesh_infrastructure(self):
        """Test basic mesh setup and dimension property."""
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        ref = _ReferenceStub(domain)
        assert ref.dim == 2

    def test_facet_operations(self):
        """Test facet retrieval and periodic meshtag generation."""
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        ref = _ReferenceStub(domain)

        facets = ref.get_facets("all")
        assert facets.size > 0

        tags = ref.periodic_meshtag(tag=7)
        assert np.all(tags.values == 7)

    def test_interpolation_onto_function_space(self):
        """Test solution interpolation onto dolfinx function space."""
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        sol = _AnalyticalStub2D()
        sol.domain = domain

        V = fem.functionspace(domain, ("Lagrange", 1))
        fn = fem.Function(V)
        sol.interpolate_field(fn, field="p", t=0.5)

        assert (
            fn.x.array.shape[0] == V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        )


# ============================================================================
# Taylor-Green 2D: Initialization, Mesh, and Periodic Boundaries
# ============================================================================


class TestTaylorGreenInitialization:
    """Test Taylor-Green initialization and parameter setting."""

    def test_default_initialization(self):
        """Test default parameter values."""
        tg = TaylorGreen2D()
        assert tg.L == 2 * np.pi
        assert tg.nu == 1
        assert tg.rho == 1
        assert tg.mu == 1

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        tg = TaylorGreen2D(L=4 * np.pi, nu=0.5, mode=2)
        assert tg.L == 4 * np.pi
        assert tg.nu == 0.5
        assert tg.mu == 0.5

    def test_mesh_creation(self):
        """Test mesh creation with default and custom resolutions."""
        tg = TaylorGreen2D(L=2 * np.pi)
        dom = tg.create_mesh()
        assert dom is not None
        assert tg.domain is dom
        assert tg.dim == 2

        # Test mesh extent
        coords = dom.geometry.x
        L = 2 * np.pi
        assert coords[:, 0].min() >= -1e-10
        assert coords[:, 0].max() <= L + 1e-10


class TestTaylorGreenPeriodicBoundaries:
    """Test periodic boundary indicator and relation."""

    def test_periodic_boundary_indicator(self):
        """Test identification of periodic boundary points."""
        L = 2 * np.pi
        tg = TaylorGreen2D(L=L)

        # Right edge (x=L)
        x_right = np.array([[L], [0.5]])
        assert tg.periodic_boundary_indicator(x_right)[0] == True

        # Top edge (y=L)
        x_top = np.array([[0.5], [L]])
        assert tg.periodic_boundary_indicator(x_top)[0] == True

        # Interior point
        x_interior = np.array([[0.5], [0.5]])
        assert tg.periodic_boundary_indicator(x_interior)[0] == False

    def test_periodic_relation(self):
        """Test mapping from boundary to opposite boundary."""
        L = 2 * np.pi
        tg = TaylorGreen2D(L=L)

        # Point at right edge maps to left edge
        x_right = np.array([[L, 0.5], [0.5, 1.0]])
        x_mapped = tg.periodic_relation(x_right)
        np.testing.assert_allclose(x_mapped[0], [0.0, 0.5], atol=1e-10)

        # Point at top edge maps to bottom edge
        x_top = np.array([[0.5, 1.0], [L, 0.5]])
        x_mapped = tg.periodic_relation(x_top)
        np.testing.assert_allclose(x_mapped[1], [0.0, 0.5], atol=1e-10)

    def test_facet_getters(self):
        """Test facet getter registration."""
        tg = TaylorGreen2D(L=2 * np.pi)
        tg.create_mesh(nx=4, ny=4)

        assert "domain_boundary" in tg.available_facet_getters
        assert "walls_vertical" in tg.available_facet_getters
        assert "walls_horizontal" in tg.available_facet_getters

        # Verify facets can be retrieved
        facets = tg.get_facets("domain_boundary")
        assert facets.size > 0


# ============================================================================
# Reference Data System Tests
# ============================================================================


class TestReferenceDataSystem:
    """Test the reference data registration and retrieval system."""

    def test_register_and_get_static_data(self):
        """Test registering and retrieving static reference data."""
        sol = ReferenceSolution()
        sol.register_reference_data("test_value", 42.0)
        sol.register_reference_data("test_tuple", (1.0, 2.0))

        assert sol.get_reference_data("test_value") == 42.0
        assert sol.get_reference_data("test_tuple") == (1.0, 2.0)

    def test_register_and_get_callable_data(self):
        """Test registering and retrieving callable reference data."""
        sol = ReferenceSolution()

        def getter_no_params():
            return np.array([1, 2, 3])

        def getter_with_params(multiplier=1):
            return np.array([1, 2, 3]) * multiplier

        sol.register_reference_data("data_static", getter_no_params)
        sol.register_reference_data("data_param", getter_with_params)

        np.testing.assert_allclose(sol.get_reference_data("data_static"), [1, 2, 3])
        np.testing.assert_allclose(
            sol.get_reference_data("data_param", multiplier=2), [2, 4, 6]
        )

    def test_available_reference_data_property(self):
        """Test available_reference_data property returns all keys."""
        sol = ReferenceSolution()
        sol.register_reference_data("static1", 10)
        sol.register_reference_data("static2", 20)
        sol.register_reference_data("callable1", lambda: 30)

        keys = sol.available_reference_data
        assert "static1" in keys
        assert "static2" in keys
        assert "callable1" in keys
        assert len(keys) == 3

    def test_get_nonexistent_key_raises_error(self):
        """Test that accessing nonexistent key raises KeyError."""
        sol = ReferenceSolution()
        sol.register_reference_data("existing", 42)

        with pytest.raises(KeyError) as excinfo:
            sol.get_reference_data("nonexistent")

        assert "nonexistent" in str(excinfo.value)
        assert "Available keys" in str(excinfo.value)
