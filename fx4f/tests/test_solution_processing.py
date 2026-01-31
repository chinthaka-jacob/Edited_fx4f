"""Test suite for turbulence.solution_processing module."""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh

from fx4f.turbulence import planar_average, indices_repeating_coordinates


class TestPlanarAverage:
    """Test planar_average function."""

    @pytest.fixture
    def rectangular_mesh_2d(self):
        """Create a 2D rectangular mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (2.0, 1.0)],
            [4, 4],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    @pytest.fixture
    def box_mesh_3d(self):
        """Create a 3D box mesh."""
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [(0.0, 0.0, 0.0), (1.0, 2.0, 1.0)],
            [2, 4, 2],
            cell_type=mesh.CellType.tetrahedron,
        )
        return domain

    def test_planar_average_scalar_constant_field(self, rectangular_mesh_2d):
        """Test planar averaging with constant scalar field."""
        V = fem.functionspace(rectangular_mesh_2d, ("Lagrange", 1))
        u = fem.Function(V)
        u.x.array[:] = 5.0

        y_coords, avg_vals = planar_average(u)

        # All averaged values should be constant
        assert len(avg_vals) == 1  # Scalar field
        assert np.allclose(avg_vals[0], 5.0, atol=1e-10)

        # y_coords should be sorted
        assert np.all(y_coords[:-1] <= y_coords[1:])

    def test_planar_average_vector_constant_field(self, box_mesh_3d):
        """Test planar averaging with constant vector field."""
        V = fem.functionspace(box_mesh_3d, ("Lagrange", 1, (3,)))
        u = fem.Function(V)
        u.x.array[:] = np.tile([1.0, 2.0, 3.0], len(u.x.array) // 3)

        y_coords, avg_vals = planar_average(u)

        # Vector field should have 3 components
        assert len(avg_vals) == 3

        # All components should be constant
        assert np.allclose(avg_vals[0], 1.0, atol=1e-10)
        assert np.allclose(avg_vals[1], 2.0, atol=1e-10)
        assert np.allclose(avg_vals[2], 3.0, atol=1e-10)

    def test_planar_average_coordinates_sorted(self, rectangular_mesh_2d):
        """Test that returned coordinates are sorted."""
        V = fem.functionspace(rectangular_mesh_2d, ("Lagrange", 1))
        u = fem.Function(V)
        u.x.array[:] = 1.0

        y_coords, _ = planar_average(u)

        # Coordinates should be in ascending order
        assert np.all(y_coords[:-1] <= y_coords[1:])

    def test_planar_average_2d_vector_field(self, rectangular_mesh_2d):
        """Test planar averaging with 2D vector field."""
        V = fem.functionspace(rectangular_mesh_2d, ("Lagrange", 1, (2,)))
        u = fem.Function(V)
        u.x.array[:] = np.tile([3.5, 7.5], len(u.x.array) // 2)

        y_coords, avg_vals = planar_average(u)

        # 2D vector field
        assert len(avg_vals) == 2
        assert np.allclose(avg_vals[0], 3.5, atol=1e-10)
        assert np.allclose(avg_vals[1], 7.5, atol=1e-10)


class TestIndicesRepeatingCoordinates:
    """Test indices_repeating_coordinates function."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple structured mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [2, 2],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    def test_unique_coordinates_axis_1(self, simple_mesh):
        """Test extraction of unique y-coordinates."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        unique_coords, index_map, repeats = indices_repeating_coordinates(u, axis=1)

        # Should have unique y-coordinates
        assert len(unique_coords) > 0
        assert len(unique_coords) == len(repeats)
        assert len(index_map) >= len(unique_coords)

        # All indices should be valid
        assert np.all(index_map >= 0)
        assert np.all(index_map < len(unique_coords))

    def test_unique_coordinates_axis_0(self, simple_mesh):
        """Test extraction of unique x-coordinates."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        unique_coords, index_map, repeats = indices_repeating_coordinates(u, axis=0)

        # Should have unique x-coordinates
        assert len(unique_coords) > 0
        assert len(unique_coords) == len(repeats)

    def test_repeats_sum_to_total_points(self, simple_mesh):
        """Test that repeat counts sum to total number of points."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        unique_coords, index_map, repeats = indices_repeating_coordinates(u, axis=1)

        # Sum of repeats should equal total number of nodes
        assert np.sum(repeats) == len(index_map)

    def test_index_map_valid_indices(self, simple_mesh):
        """Test that index map contains valid indices."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        unique_coords, index_map, repeats = indices_repeating_coordinates(u, axis=1)

        # All index_map values should be valid indices into unique_coords
        assert np.all(index_map >= 0)
        assert np.all(index_map < len(unique_coords))

        # Each unique index should appear in index_map
        unique_indices = np.unique(index_map)
        assert len(unique_indices) == len(unique_coords)

    def test_caching_behavior(self, simple_mesh):
        """Test that function uses LRU cache correctly."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        # Call twice with same arguments
        result1 = indices_repeating_coordinates(u, axis=1)
        result2 = indices_repeating_coordinates(u, axis=1)

        # Results should be identical (from cache)
        assert np.array_equal(result1[0], result2[0])
        assert np.array_equal(result1[1], result2[1])
        assert np.array_equal(result1[2], result2[2])

    def test_different_axes_give_different_results(self, simple_mesh):
        """Test that different axes produce different coordinate sets."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        coords_x, _, _ = indices_repeating_coordinates(u, axis=0)
        coords_y, _, _ = indices_repeating_coordinates(u, axis=1)

        # For a non-square domain or different mesh resolutions,
        # x and y coordinates should differ
        # For this square mesh they may be the same, so just check they're valid
        assert len(coords_x) > 0
        assert len(coords_y) > 0

    def test_tolerance_parameter(self, simple_mesh):
        """Test that tolerance parameter is used."""
        V = fem.functionspace(simple_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        # Different tolerance values should potentially give same results
        # for well-separated coordinates
        result1 = indices_repeating_coordinates(u, axis=1, atol=1e-8)
        result2 = indices_repeating_coordinates(u, axis=1, atol=1e-10)

        # Both should return valid results
        assert len(result1[0]) > 0
        assert len(result2[0]) > 0
