"""Test suite for fluid_mechanics.initial_conditions module.

Note: Full integration tests with mixed function spaces should be added
by the user. These tests validate the function can be called and produces
valid output.
"""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
from basix.ufl import element, mixed_element

from fx4f.fluid_mechanics import compute_dudt_p_from_u


class TestComputeDudtPFromU:
    """Test compute_dudt_p_from_u function."""

    @pytest.fixture
    def setup_mixed_space(self):
        """Create a 2D mesh with mixed velocity-pressure function space."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [(0, 0), (1, 1)], [4, 4], cell_type=mesh.CellType.triangle
        )

        # Create mixed element (velocity P2, pressure P1)
        P2 = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
        P1 = element("Lagrange", domain.basix_cell(), 1)
        We = mixed_element([P2, P1])
        W = fem.functionspace(domain, We)

        return domain, W

    def test_constant_velocity_returns_zero(self, setup_mixed_space):
        """Test with constant velocity field."""
        domain, W = setup_mixed_space
        V = fem.functionspace(domain, ("Lagrange", 2, (2,)))

        u = fem.Function(V)
        u.x.array[:] = 1.0

        result = compute_dudt_p_from_u(W, u, mu=1.0, rho=1.0)

        assert isinstance(result, fem.Function)
        assert result.function_space == W
        assert np.allclose(result.x.array, 0.0, atol=1e-12)

    def test_with_stabilization(self, setup_mixed_space):
        """Test with stabilization enabled."""
        domain, W = setup_mixed_space
        V = fem.functionspace(domain, ("Lagrange", 2, (2,)))

        u = fem.Function(V)
        u.x.array[:] = 1.0

        result = compute_dudt_p_from_u(W, u, mu=1.0, rho=1.0, stabilization=True)

        assert isinstance(result, fem.Function)
        assert result.function_space == W
        assert np.all(np.isfinite(result.x.array))
