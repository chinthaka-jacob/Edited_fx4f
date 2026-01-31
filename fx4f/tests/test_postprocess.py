"""Test suite for fluid_mechanics.postprocess module."""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
from ufl import dx

from fx4f.fluid_mechanics import zero_mean_pressure


class TestZeroMeanPressure:
    """Basic sanity checks for pressure post-processing."""

    @pytest.fixture
    def pressure_space(self):
        """Create a 2D mesh with pressure function space."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [(0, 0), (1, 1)], [4, 4], cell_type=mesh.CellType.triangle
        )
        Q = fem.functionspace(domain, ("Lagrange", 1))
        return domain, Q

    def test_zero_field_remains_zero(self, pressure_space):
        """Test that zero pressure field remains zero."""
        domain, Q = pressure_space
        p = fem.Function(Q)
        p.x.array[:] = 0.0

        zero_mean_pressure(p)

        assert np.allclose(p.x.array, 0.0, atol=1e-12)

    def test_constant_field_becomes_zero(self, pressure_space):
        """Test that constant pressure field is zeroed."""
        domain, Q = pressure_space
        p = fem.Function(Q)
        p.x.array[:] = 5.0

        zero_mean_pressure(p)

        assert np.allclose(p.x.array, 0.0, atol=1e-10)

    def test_mean_becomes_zero(self, pressure_space):
        """Test that mean of pressure field becomes zero."""
        domain, Q = pressure_space
        p = fem.Function(Q)

        # Linear field
        def linear(x):
            return 2.0 * x[0] + 1.0

        p.interpolate(linear)
        zero_mean_pressure(p)

        # Verify mean is approximately zero
        vol = domain.comm.allreduce(
            fem.assemble_scalar(fem.form(fem.Constant(domain, 1.0) * dx)),
            op=MPI.SUM,
        )
        p_integral = fem.assemble_scalar(fem.form(p * dx))
        p_mean = domain.comm.allreduce(p_integral, op=MPI.SUM) / vol

        assert np.abs(p_mean) < 1e-10

    def test_idempotent(self, pressure_space):
        """Test that applying zero mean pressure twice gives same result."""
        domain, Q = pressure_space
        p = fem.Function(Q)

        def linear(x):
            return x[0] + 2 * x[1]

        p.interpolate(linear)

        zero_mean_pressure(p)
        p_once = p.x.array.copy()

        zero_mean_pressure(p)
        p_twice = p.x.array.copy()

        assert np.allclose(p_once, p_twice, atol=1e-14)

    def test_preserves_differences(self, pressure_space):
        """Test that pressure differences are preserved."""
        domain, Q = pressure_space
        p = fem.Function(Q)

        def field(x):
            return x[0] + 2 * x[1]

        p.interpolate(field)
        p_orig = p.x.array.copy()
        p_range_orig = np.max(p_orig) - np.min(p_orig)

        zero_mean_pressure(p)
        p_range = np.max(p.x.array) - np.min(p.x.array)

        # Range should be the same (just shifted)
        assert np.isclose(p_range_orig, p_range, rtol=1e-10)
