"""Tests for miscellaneous.projections module.

Tests verify L2 and H1 projection operators with fem.Function inputs.

Note: Current implementation requires inputs to be fem.Function or UFL expressions,
not Python callables. Tests use interpolated functions as input.
"""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
import ufl

from fx4f.miscellaneous import L2_projection, H1_projection


def _compute_L2_norm(u: fem.Function) -> float:
    """Compute L2 norm of a function."""
    L2_form = fem.form(ufl.inner(u, u) * ufl.dx)
    local_norm = fem.assemble_scalar(L2_form)
    global_norm_sq = u.function_space.mesh.comm.allreduce(local_norm, op=MPI.SUM)
    return np.sqrt(global_norm_sq)


class TestL2Projection:
    """Test suite for L2 projection."""

    @pytest.fixture
    def unit_square_mesh(self):
        """Create a simple 2D unit square mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [8, 8],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    def test_L2_projection_from_function(self, unit_square_mesh):
        """Test L2 projection from another fem.Function."""
        V_source = fem.functionspace(unit_square_mesh, ("Lagrange", 3))
        V_target = fem.functionspace(unit_square_mesh, ("Lagrange", 2))

        # Create source function
        u_source = fem.Function(V_source)
        u_source.interpolate(lambda x: x[0] ** 3 + x[1] ** 3)

        # Project to lower-order space
        u_proj = L2_projection(V_target, u_source)

        assert isinstance(u_proj, fem.Function), "Should return fem.Function"
        assert u_proj.function_space == V_target, "Should be in target space"
        assert np.all(np.isfinite(u_proj.x.array[:])), "Projection should be finite"

    def test_L2_projection_polynomial_exact(self, unit_square_mesh):
        """Test that L2 projection is exact for polynomials in space."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))

        # Create a quadratic function (exactly in P2 space)
        u_source = fem.Function(V)
        u_source.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

        # Project onto same space
        u_proj = L2_projection(V, u_source)

        # Should be very close to original
        diff = u_proj.x.array[:] - u_source.x.array[:]
        assert np.allclose(
            diff, 0.0, atol=1e-8
        ), "Polynomial in space should project exactly"

    def test_L2_projection_vector_function(self, unit_square_mesh):
        """Test L2 projection of vector-valued function."""
        V_source = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))
        V_target = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))

        u_source = fem.Function(V_source)
        u_source.interpolate(lambda x: np.vstack([x[0] ** 2, x[1] ** 2]))

        u_proj = L2_projection(V_target, u_source)

        # Check return type and shape
        assert isinstance(u_proj, fem.Function), "Should return fem.Function"
        assert u_proj.function_space == V_target, "Should be in correct function space"
        assert np.all(np.isfinite(u_proj.x.array[:])), "Projection should be finite"

    def test_L2_projection_different_degrees(self):
        """Test L2 projection between spaces of different polynomial degrees."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [10, 10],
            cell_type=mesh.CellType.triangle,
        )
        V_high = fem.functionspace(domain, ("Lagrange", 4))
        V_low = fem.functionspace(domain, ("Lagrange", 1))

        # Create high-order function
        u_high = fem.Function(V_high)
        u_high.interpolate(
            lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
        )

        # Project to low-order space
        u_low = L2_projection(V_low, u_high)

        assert isinstance(u_low, fem.Function), "Should return fem.Function"
        assert u_low.function_space == V_low, "Should be in target space"

        # Projection should have bounded norm
        norm_low = _compute_L2_norm(u_low)
        assert (
            0 < norm_low < 1.0
        ), f"Projected norm should be reasonable, got {norm_low}"

    def test_L2_projection_clamp_not_implemented(self, unit_square_mesh):
        """Test that clamp option raises NotImplementedError."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))
        u = fem.Function(V)
        u.interpolate(lambda x: np.full(x.shape[1], 1.0))

        with pytest.raises(NotImplementedError):
            L2_projection(V, u, clamp=True)


class TestH1Projection:
    """Test suite for H1 projection."""

    @pytest.fixture
    def unit_square_mesh(self):
        """Create a simple 2D unit square mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [8, 8],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    def test_H1_projection_from_function(self, unit_square_mesh):
        """Test H1 projection from another fem.Function."""
        V_source = fem.functionspace(unit_square_mesh, ("Lagrange", 3))
        V_target = fem.functionspace(unit_square_mesh, ("Lagrange", 2))

        u_source = fem.Function(V_source)
        u_source.interpolate(lambda x: x[0] ** 2 + 2.0 * x[1] ** 2)

        u_proj = H1_projection(V_target, u_source)

        assert isinstance(u_proj, fem.Function), "Should return fem.Function"
        assert u_proj.function_space == V_target, "Should be in target space"
        assert np.all(np.isfinite(u_proj.x.array[:])), "H1 projection should be finite"

    def test_H1_projection_polynomial_exact(self, unit_square_mesh):
        """Test that H1 projection is exact for polynomials in space."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))

        # Create a quadratic function
        u_source = fem.Function(V)
        u_source.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

        u_proj = H1_projection(V, u_source)

        # Should be very close to original
        diff = u_proj.x.array[:] - u_source.x.array[:]
        assert np.allclose(
            diff, 0.0, atol=1e-8
        ), "Polynomial in space should H1-project exactly"

    def test_H1_projection_L2_weight(self, unit_square_mesh):
        """Test H1 projection with different L2 weights."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 3))

        u_source = fem.Function(V)
        u_source.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

        # Project with different L2 weights
        u_proj_w1 = H1_projection(V, u_source, L2_weight=1.0)
        u_proj_w10 = H1_projection(V, u_source, L2_weight=10.0)

        # Both should return valid functions
        assert isinstance(u_proj_w1, fem.Function), "Should return fem.Function"
        assert isinstance(u_proj_w10, fem.Function), "Should return fem.Function"
        assert np.all(np.isfinite(u_proj_w1.x.array[:])), "Projection should be finite"
        assert np.all(np.isfinite(u_proj_w10.x.array[:])), "Projection should be finite"

    def test_H1_projection_vector_function(self, unit_square_mesh):
        """Test H1 projection of vector-valued function."""
        V_source = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))
        V_target = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))

        u_source = fem.Function(V_source)
        u_source.interpolate(
            lambda x: np.vstack([np.sin(np.pi * x[0]), np.cos(np.pi * x[1])])
        )

        u_proj = H1_projection(V_target, u_source)

        # Check return type and shape
        assert isinstance(u_proj, fem.Function), "Should return fem.Function"
        assert u_proj.function_space == V_target, "Should be in correct function space"
        assert np.all(np.isfinite(u_proj.x.array[:])), "H1 projection should be finite"

    def test_H1_projection_clamp_not_implemented(self, unit_square_mesh):
        """Test that clamp option raises NotImplementedError."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))
        u = fem.Function(V)
        u.interpolate(lambda x: np.full(x.shape[1], 1.0))

        with pytest.raises(NotImplementedError):
            H1_projection(V, u, clamp=True)
