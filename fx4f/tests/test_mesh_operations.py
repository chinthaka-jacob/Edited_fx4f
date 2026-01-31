"""Tests for miscellaneous.mesh_operations module.

Tests verify mesh operation utilities including domain volume computation,
element size calculation, point evaluation, and MPI-aware coefficient gathering.
"""

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh

from fx4f.miscellaneous import (
    get_domain_volume,
    get_element_sizes,
    get_point_values,
    gather_coefficients,
)


class TestMeshOperations:
    """Test suite for mesh operation utilities."""

    NX = 4  # Number of elements in x-direction for mesh tests

    @pytest.fixture
    def unit_square_mesh(self):
        """Create a simple 2D unit square mesh."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (1.0, 1.0)],
            [self.NX, self.NX],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    @pytest.fixture
    def unit_cube_mesh(self):
        """Create a simple 3D unit cube mesh."""
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [self.NX, self.NX, self.NX],
            cell_type=mesh.CellType.tetrahedron,
        )
        return domain

    @pytest.fixture
    def rectangle_mesh(self):
        """Create a 2D rectangle mesh with known area."""
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (2.0, 3.0)],
            [6, 8],
            cell_type=mesh.CellType.triangle,
        )
        return domain

    def test_get_domain_volume_unit_square(self, unit_square_mesh):
        """Test domain volume computation on unit square."""
        vol = get_domain_volume(unit_square_mesh)
        assert np.isclose(vol, 1.0, rtol=1e-10), f"Expected volume 1.0, got {vol}"

    def test_get_domain_volume_unit_cube(self, unit_cube_mesh):
        """Test domain volume computation on unit cube."""
        vol = get_domain_volume(unit_cube_mesh)
        assert np.isclose(vol, 1.0, rtol=1e-10), f"Expected volume 1.0, got {vol}"

    def test_get_domain_volume_rectangle(self, rectangle_mesh):
        """Test domain volume computation on rectangle with known area."""
        vol = get_domain_volume(rectangle_mesh)
        expected_area = 2.0 * 3.0  # 6.0
        assert np.isclose(
            vol, expected_area, rtol=1e-10
        ), f"Expected volume {expected_area}, got {vol}"

    def test_get_domain_volume_caching(self, unit_square_mesh):
        """Test that get_domain_volume caching works correctly."""
        vol1 = get_domain_volume(unit_square_mesh)
        vol2 = get_domain_volume(unit_square_mesh)
        # Should return identical results due to caching
        assert vol1 == vol2, "Cached results should be identical"
        assert np.isclose(vol1, 1.0, rtol=1e-10)

    @pytest.mark.parametrize("nx,ny", [(2, 2), (2, 4), (2, 8)])
    def test_get_element_sizes_2d(self, nx, ny):
        """Test element size computation on 2D meshes with varying resolution."""
        L, H = 1.0, 3.0
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), (L, H)],
            [nx, ny],
            cell_type=mesh.CellType.triangle,
        )
        h = get_element_sizes(domain)

        # Check return type
        assert isinstance(h, fem.Function), "Should return fem.Function"

        # Check function space is DG0
        assert (
            h.function_space.element.basix_element.discontinuous
        ), "Should be DG space"
        assert h.function_space.element.basix_element.degree == 0, "Should be degree 0"

        # Check all values are positive and finite
        assert np.all(h.x.array[:] > 0), "Element sizes should be positive"
        assert np.all(np.isfinite(h.x.array[:])), "Element sizes should be finite"

        # Expected element size: approximately sqrt(element_area)
        # For unit square divided into nx*ny*2 triangles, each triangle has area ~= 1/(2*nx*ny)
        expected_h = np.sqrt(L * H / (2.0 * nx * ny))
        mean_h = np.mean(h.x.array[:])
        assert np.isclose(
            mean_h, expected_h, rtol=1e-10
        ), f"Expected mean element size ~{expected_h}, got {mean_h}"

    def test_get_element_sizes_3d(self, unit_cube_mesh):
        """Test element size computation on 3D mesh."""
        h = get_element_sizes(unit_cube_mesh)

        # Check return type
        assert isinstance(h, fem.Function), "Should return fem.Function"

        # Check all values are positive and finite
        assert np.all(h.x.array[:] > 0), "Element sizes should be positive"
        assert np.all(np.isfinite(h.x.array[:])), "Element sizes should be finite"

        expected_h = (1.0 / len(h.x.array[:])) ** (1 / 3)
        mean_h = np.mean(h.x.array[:])
        assert np.isclose(
            mean_h, expected_h, rtol=1e-10
        ), f"Expected mean element size ~{expected_h}, got {mean_h}"

    def test_get_point_values_constant_function(self, unit_square_mesh):
        """Test point evaluation with constant function."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
        u = fem.Function(V)
        u.x.array[:] = 5.0  # Constant value

        points = [
            (0.25, 0.25, 0.0),
            (0.5, 0.5, 0.0),
            (0.75, 0.75, 0.0),
        ]

        values = get_point_values(u, points)

        assert len(values) == len(points), "Should return value for each point"
        for val in values:
            assert np.isclose(val, 5.0, rtol=1e-10), f"Expected 5.0, got {val}"

    def test_get_point_values_linear_function(self, unit_square_mesh):
        """Test point evaluation with linear polynomial."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        # Set u = x + 2*y
        def linear_func(x):
            return x[0] + 2.0 * x[1]

        u.interpolate(linear_func)

        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
        ]

        expected_values = [0.0, 1.0, 2.0, 1.5]

        values = get_point_values(u, points)

        for val, expected in zip(values, expected_values):
            assert np.isclose(
                val, expected, atol=1e-10
            ), f"Expected {expected}, got {val}"

    def test_get_point_values_vector_function(self, unit_square_mesh):
        """Test point evaluation with vector-valued function."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))
        u = fem.Function(V)

        # Set u = [x, y]
        def vector_func(x):
            return np.vstack([x[0], x[1]])

        u.interpolate(vector_func)

        points = [(0.3, 0.7, 0.0)]
        values = get_point_values(u, points)

        assert len(values) == 1
        assert len(values[0]) == 2, "Vector function should return 2 components"
        assert np.isclose(values[0][0], 0.3, atol=1e-10)
        assert np.isclose(values[0][1], 0.7, atol=1e-10)

    def test_gather_coefficients_scalar_function(self, unit_square_mesh):
        """Test coefficient gathering for scalar function."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
        u = fem.Function(V)

        # Set a simple pattern
        def simple_func(x):
            return x[0] + x[1]

        u.interpolate(simple_func)

        coeffs = gather_coefficients(u)

        if MPI.COMM_WORLD.rank == 0:
            # Check coefficients are gathered
            assert len(coeffs) > 0, "Should have gathered coefficients on rank 0"
            assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        else:
            # On other ranks, should return minimal array
            assert len(coeffs) == 1, "Non-root ranks should return minimal array"

    def test_gather_coefficients_vector_function(self, unit_square_mesh):
        """Test coefficient gathering for vector function."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 1, (2,)))
        u = fem.Function(V)

        def vector_func(x):
            return np.vstack([x[0], x[1]])

        u.interpolate(vector_func)

        coeffs = gather_coefficients(u)

        if MPI.COMM_WORLD.rank == 0:
            assert len(coeffs) > 0, "Should have gathered coefficients on rank 0"
            assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        else:
            # For vector function with 2 components
            assert len(coeffs) == 2, "Non-root ranks should return array of size 2"

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_get_point_values_polynomial_accuracy(self, unit_square_mesh, degree):
        """Test point evaluation accuracy with different polynomial degrees."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", degree))
        u = fem.Function(V)

        # Use polynomial that should be exactly represented
        # For degree d, use polynomial of degree d
        if degree == 1:

            def poly_func(x):
                return x[0] + 2.0 * x[1]

            test_point = (0.3, 0.4, 0.0)
            expected = 0.3 + 2.0 * 0.4
        elif degree == 2:

            def poly_func(x):
                return x[0] ** 2 + x[1] ** 2

            test_point = (0.5, 0.5, 0.0)
            expected = 0.5**2 + 0.5**2
        else:  # degree == 3

            def poly_func(x):
                return x[0] ** 3 + x[1] ** 2

            test_point = (0.4, 0.6, 0.0)
            expected = 0.4**3 + 0.6**2

        u.interpolate(poly_func)
        values = get_point_values(u, [test_point])

        assert np.isclose(
            values[0], expected, atol=1e-9
        ), f"Expected {expected}, got {values[0]} for degree {degree}"

    def test_sample_on_refined_linears_no_refinement(self, unit_square_mesh):
        """Test interpolation to P1 without refinement (refinement_level=0)."""
        # Create a higher-order field
        V_high = fem.functionspace(unit_square_mesh, ("Lagrange", 3))
        u_high = fem.Function(V_high)
        u_high.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)
        u_high.name = "test_field"

        # Sample on P1 without refinement
        from fx4f.miscellaneous import sample_on_refined_linears

        u_p1_list = sample_on_refined_linears(u_high, refinement_level=0)

        # Check return type
        assert isinstance(u_p1_list, list), "Should return list"
        assert len(u_p1_list) == 1, "Should have one function"

        u_p1 = u_p1_list[0]

        # Check properties
        assert isinstance(u_p1, fem.Function), "Should return fem.Function"
        assert u_p1.name == "test_field", "Should preserve function name"

        # Check that result is P1
        assert u_p1.function_space.element.basix_element.degree == 1, "Should be P1"

        # Check result is finite
        assert np.all(np.isfinite(u_p1.x.array[:])), "Result should be finite"

    def test_sample_on_refined_linears_multiple_fields(self, unit_square_mesh):
        """Test interpolation of multiple fields to P1."""
        V_source = fem.functionspace(unit_square_mesh, ("Lagrange", 2, (2,)))
        u_vec = fem.Function(V_source)
        u_vec.interpolate(lambda x: np.vstack([x[0] ** 2, x[1] ** 2]))
        u_vec.name = "vector_field"

        V_scalar = fem.functionspace(unit_square_mesh, ("Lagrange", 3))
        u_scalar = fem.Function(V_scalar)
        u_scalar.interpolate(lambda x: x[0] * x[1])
        u_scalar.name = "scalar_field"

        # Sample multiple fields on P1
        from fx4f.miscellaneous import sample_on_refined_linears

        u_p1_list = sample_on_refined_linears([u_vec, u_scalar], refinement_level=0)

        # Check we got both functions back
        assert len(u_p1_list) == 2, "Should have two functions"
        assert (
            u_p1_list[0].name == "vector_field"
        ), "First field name should be preserved"
        assert (
            u_p1_list[1].name == "scalar_field"
        ), "Second field name should be preserved"

        # Check they're both P1
        for u in u_p1_list:
            assert u.function_space.element.basix_element.degree == 1, "Should be P1"
            assert np.all(np.isfinite(u.x.array[:])), "Results should be finite"

    def test_sample_on_refined_linears_refinement_not_implemented(
        self, unit_square_mesh
    ):
        """Test that mesh refinement (refinement_level > 0) raises NotImplementedError."""
        V = fem.functionspace(unit_square_mesh, ("Lagrange", 2))
        u = fem.Function(V)
        u.interpolate(lambda x: x[0] + x[1])

        from fx4f.miscellaneous import sample_on_refined_linears

        with pytest.raises(NotImplementedError):
            sample_on_refined_linears(u, refinement_level=1)
