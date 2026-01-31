"""Test suite for turbulence.statistics module."""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
from basix.ufl import element

from fx4f.turbulence import TurbulenceStatistics


class TestTurbulenceStatistics:
    """Test TurbulenceStatistics class."""

    @pytest.fixture
    def unit_square_mesh(self):
        """Create a simple 2D mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [4, 4],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    @pytest.fixture
    def unit_cube_mesh(self):
        """Create a simple 3D mesh."""
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [3, 3, 3],
            cell_type=mesh.CellType.tetrahedron,
        )
        return domain

    @pytest.fixture
    def scalar_function_space(self, unit_square_mesh):
        """Create a scalar P2 function space."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))
        return V

    @pytest.fixture
    def vector_function_space(self, unit_square_mesh):
        """Create a vector P2 function space."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))
        return V

    def test_mean_scalar_constant_fields(self, scalar_function_space):
        """Test mean calculation with constant scalar fields."""
        stats = TurbulenceStatistics("mean")

        # Create three constant fields with values 1, 2, 3
        u1 = fem.Function(scalar_function_space)
        u1.x.array[:] = 1.0

        u2 = fem.Function(scalar_function_space)
        u2.x.array[:] = 2.0

        u3 = fem.Function(scalar_function_space)
        u3.x.array[:] = 3.0

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        mean = stats.mean

        # Mean should be (1 + 2 + 3) / 3 = 2.0
        assert np.allclose(mean.x.array, 2.0, atol=1e-12)

    def test_mean_vector_constant_fields(self, vector_function_space):
        """Test mean calculation with constant vector fields."""
        stats = TurbulenceStatistics("mean")

        # Create constant vector fields
        u1 = fem.Function(vector_function_space)
        u1.x.array[:] = np.tile([1.0, 0.5], len(u1.x.array) // 2)

        u2 = fem.Function(vector_function_space)
        u2.x.array[:] = np.tile([2.0, 1.5], len(u2.x.array) // 2)

        u3 = fem.Function(vector_function_space)
        u3.x.array[:] = np.tile([3.0, 2.5], len(u3.x.array) // 2)

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        mean = stats.mean
        expected = np.tile([2.0, 1.5], len(mean.x.array) // 2)

        assert np.allclose(mean.x.array, expected, atol=1e-12)

    def test_rms_zero_fluctuations(self, scalar_function_space):
        """Test RMS with identical fields (zero fluctuations)."""
        stats = TurbulenceStatistics("mean", "rms")

        # All fields are identical -> zero fluctuations
        u1 = fem.Function(scalar_function_space)
        u1.x.array[:] = 2.5

        u2 = fem.Function(scalar_function_space)
        u2.x.array[:] = 2.5

        u3 = fem.Function(scalar_function_space)
        u3.x.array[:] = 2.5

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        rms = stats.rms
        rms2 = stats.rms2

        # RMS should be zero when all fields are identical
        assert np.allclose(rms.x.array, 0.0, atol=1e-10)
        assert np.allclose(rms2.x.array, 0.0, atol=1e-10)

    def test_rms_simple_fluctuations(self, scalar_function_space):
        """Test RMS calculation with simple fluctuations."""
        stats = TurbulenceStatistics("mean", "rms")

        # Fields: 0, 2, 4 -> mean = 2, fluctuations: -2, 0, 2
        # rms2 = ((−2)² + 0² + 2²) / 3 = 8/3
        # rms = sqrt(8/3) ≈ 1.6329931618554521
        u1 = fem.Function(scalar_function_space)
        u1.x.array[:] = 0.0

        u2 = fem.Function(scalar_function_space)
        u2.x.array[:] = 2.0

        u3 = fem.Function(scalar_function_space)
        u3.x.array[:] = 4.0

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        mean = stats.mean
        rms2 = stats.rms2
        rms = stats.rms

        expected_mean = 2.0
        expected_rms2 = 8.0 / 3.0
        expected_rms = np.sqrt(8.0 / 3.0)

        assert np.allclose(mean.x.array, expected_mean, atol=1e-12)
        assert np.allclose(rms2.x.array, expected_rms2, atol=1e-10)
        assert np.allclose(rms.x.array, expected_rms, atol=1e-10)

    def test_rms_vector_fields(self, vector_function_space):
        """Test RMS calculation with vector fields."""
        stats = TurbulenceStatistics("mean", "rms")

        # Component-wise: [1, 2], [3, 4], [5, 6]
        # Mean: [3, 4]
        # Fluctuations: [-2, -2], [0, 0], [2, 2]
        # RMS2: [8/3, 8/3]
        u1 = fem.Function(vector_function_space)
        u1.x.array[:] = np.tile([1.0, 2.0], len(u1.x.array) // 2)

        u2 = fem.Function(vector_function_space)
        u2.x.array[:] = np.tile([3.0, 4.0], len(u2.x.array) // 2)

        u3 = fem.Function(vector_function_space)
        u3.x.array[:] = np.tile([5.0, 6.0], len(u3.x.array) // 2)

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        mean = stats.mean
        rms2 = stats.rms2

        expected_mean = np.tile([3.0, 4.0], len(mean.x.array) // 2)
        expected_rms2 = np.tile([8.0 / 3.0, 8.0 / 3.0], len(rms2.x.array) // 2)

        assert np.allclose(mean.x.array, expected_mean, atol=1e-12)
        assert np.allclose(rms2.x.array, expected_rms2, atol=1e-10)

    def test_polynomial_degree_elevation_for_rms(self, unit_square_mesh):
        """Test that RMS uses elevated polynomial degree (2x)."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
        stats = TurbulenceStatistics("rms")

        u1 = fem.Function(V)
        u1.x.array[:] = 1.0

        stats.add_frame(u1)

        # Original space has degree 1
        assert u1.function_space.ufl_element().degree == 1

        # RMS space should have degree 2 (1 * 2)
        assert stats.u_rms.function_space.ufl_element().degree == 2
        assert stats.u_rms2.function_space.ufl_element().degree == 2

    def test_single_frame_statistics(self, scalar_function_space):
        """Test statistics computation with a single frame."""
        stats = TurbulenceStatistics("mean", "rms")

        u1 = fem.Function(scalar_function_space)
        u1.x.array[:] = 3.5

        stats.add_frame(u1)

        mean = stats.mean
        rms2 = stats.rms2

        # With single frame: mean = value, rms2 = 0
        assert np.allclose(mean.x.array, 3.5, atol=1e-12)
        assert np.allclose(rms2.x.array, 0.0, atol=1e-10)

    def test_incremental_frame_addition(self, scalar_function_space):
        """Test that statistics update correctly with incremental frames."""
        stats = TurbulenceStatistics("mean")

        u1 = fem.Function(scalar_function_space)
        u1.x.array[:] = 1.0

        stats.add_frame(u1)
        mean_after_1 = stats.mean.x.array.copy()
        assert np.allclose(mean_after_1, 1.0, atol=1e-12)

        u2 = fem.Function(scalar_function_space)
        u2.x.array[:] = 3.0

        stats.add_frame(u2)
        mean_after_2 = stats.mean.x.array.copy()
        assert np.allclose(mean_after_2, 2.0, atol=1e-12)

        u3 = fem.Function(scalar_function_space)
        u3.x.array[:] = 5.0

        stats.add_frame(u3)
        mean_after_3 = stats.mean.x.array.copy()
        assert np.allclose(mean_after_3, 3.0, atol=1e-12)

    def test_many_frames(self, scalar_function_space):
        """Test statistics with many frames."""
        stats = TurbulenceStatistics("mean", "rms")

        n_frames = 10
        values = np.arange(n_frames, dtype=float)

        for val in values:
            u = fem.Function(scalar_function_space)
            u.x.array[:] = val
            stats.add_frame(u)

        mean = stats.mean
        expected_mean = np.mean(values)
        expected_variance = np.var(values)

        assert np.allclose(mean.x.array, expected_mean, atol=1e-10)
        assert np.allclose(stats.rms2.x.array, expected_variance, atol=1e-10)

    def test_3d_scalar_field(self, unit_cube_mesh):
        """Test statistics with 3D scalar fields."""
        V = fem.functionspace(unit_cube_mesh, ("Lagrange", 1))
        stats = TurbulenceStatistics("mean", "rms")

        u1 = fem.Function(V)
        u1.x.array[:] = 2.0

        u2 = fem.Function(V)
        u2.x.array[:] = 4.0

        u3 = fem.Function(V)
        u3.x.array[:] = 6.0

        stats.add_frame(u1)
        stats.add_frame(u2)
        stats.add_frame(u3)

        mean = stats.mean
        rms2 = stats.rms2

        # Mean: (2 + 4 + 6) / 3 = 4
        # Fluctuations: -2, 0, 2
        # RMS2: (4 + 0 + 4) / 3 = 8/3
        assert np.allclose(mean.x.array, 4.0, atol=1e-12)
        assert np.allclose(rms2.x.array, 8.0 / 3.0, atol=1e-10)

    def test_3d_vector_field(self, unit_cube_mesh):
        """Test statistics with 3D vector fields."""
        V = fem.functionspace(unit_cube_mesh, ("Lagrange", 1, (3,)))
        stats = TurbulenceStatistics("mean")

        u1 = fem.Function(V)
        u1.x.array[:] = np.tile([1.0, 2.0, 3.0], len(u1.x.array) // 3)

        u2 = fem.Function(V)
        u2.x.array[:] = np.tile([2.0, 4.0, 6.0], len(u2.x.array) // 3)

        stats.add_frame(u1)
        stats.add_frame(u2)

        mean = stats.mean
        expected = np.tile([1.5, 3.0, 4.5], len(mean.x.array) // 3)

        assert np.allclose(mean.x.array, expected, atol=1e-12)
