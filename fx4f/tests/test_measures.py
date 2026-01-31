"""Test suite for fluid_mechanics.measures module."""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
from ufl import ds

from fx4f.fluid_mechanics import (
    kinetic_energy,
    dissipation_rate,
    aerodynamic_forces_2D,
)


class TestMeasures:
    """Basic sanity checks for measures module."""

    @pytest.fixture
    def simple_mesh_and_space(self):
        """Create a simple 2D mesh and velocity function space."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [(0, 0), (1, 1)], [4, 4], cell_type=mesh.CellType.triangle
        )
        V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
        Q = fem.functionspace(domain, ("Lagrange", 1))
        return domain, V, Q

    def test_kinetic_energy_zero_velocity(self, simple_mesh_and_space):
        """Kinetic energy should be zero for zero velocity."""
        domain, V, Q = simple_mesh_and_space
        u = fem.Function(V)
        u.x.array[:] = 0.0

        E_kin = kinetic_energy(u)
        assert np.isclose(E_kin, 0.0, atol=1e-12)

    def test_kinetic_energy_scales_with_density(self, simple_mesh_and_space):
        """Kinetic energy should scale linearly with density."""
        domain, V, Q = simple_mesh_and_space
        u = fem.Function(V)
        u.x.array[:] = 1.0

        E_kin_rho1 = kinetic_energy(u, rho=1.0)
        E_kin_rho2 = kinetic_energy(u, rho=2.0)

        assert np.isclose(E_kin_rho2, 2.0 * E_kin_rho1, rtol=1e-10)

    def test_dissipation_rate_zero_velocity(self, simple_mesh_and_space):
        """Dissipation rate should be zero for zero velocity."""
        domain, V, Q = simple_mesh_and_space
        u = fem.Function(V)
        u.x.array[:] = 0.0

        D = dissipation_rate(u, mu=1.0)
        assert np.isclose(D, 0.0, atol=1e-12)

    def test_dissipation_rate_positive(self, simple_mesh_and_space):
        """Dissipation rate should be non-negative for any velocity field."""
        domain, V, Q = simple_mesh_and_space
        u = fem.Function(V)
        u.x.array[:] = 1.0

        D = dissipation_rate(u, mu=1.0)
        assert D >= 0.0

    def test_aerodynamic_forces_zero_fields(self, simple_mesh_and_space):
        """Aerodynamic forces should be zero for zero velocity and pressure."""
        domain, V, Q = simple_mesh_and_space
        u = fem.Function(V)
        p = fem.Function(Q)
        u.x.array[:] = 0.0
        p.x.array[:] = 0.0

        Fx, Fy = aerodynamic_forces_2D(u, p, ds, mu=1.0)

        assert np.isclose(Fx, 0.0, atol=1e-10)
        assert np.isclose(Fy, 0.0, atol=1e-10)
