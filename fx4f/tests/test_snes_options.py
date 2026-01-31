"""
Lightweight tests for SNES solver options.

Focus:
- Code coverage: every line runs
- Consistency: iteration counts match expected values
- Options verification: all PETSc options are set
"""

import pytest
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import ufl

from fx4f.solvers.snes_options import (
    set_snes_options_newtonls,
    SNESSetterRegistry,
)


@pytest.fixture
def nonlinear_poisson():
    """Create nonlinear problem: -Δu + u³ + exp(0.5u) = f."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(domain, ("Lagrange", 1))

    u = fem.Function(V)
    v = ufl.TestFunction(V)

    # Nonlinear weak form with moderate nonlinearity
    f = fem.Constant(domain, PETSc.ScalarType(3.0))
    F = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + u**3 * v * ufl.dx
        + ufl.exp(0.5 * u) * v * ufl.dx
        - f * v * ufl.dx
    )

    # Homogeneous Dirichlet BC
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0

    def boundary(x):
        return np.logical_or.reduce(
            [
                np.isclose(x[0], 0),
                np.isclose(x[0], 1),
                np.isclose(x[1], 0),
                np.isclose(x[1], 1),
            ]
        )

    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    return domain, V, u, F, [bc]


def solve_nonlinear(snes, u, F, bcs):
    """Solve nonlinear problem using SNES."""
    from dolfinx.fem.petsc import create_vector, create_matrix

    V = u.function_space
    b = create_vector(V)
    J = ufl.derivative(F, u)
    A = create_matrix(fem.form(J))

    def compute_F(snes, x, F_vec):
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()
        with F_vec.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F_vec, fem.form(F))
        fem.petsc.apply_lifting(F_vec, [fem.form(J)], [bcs])
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F_vec, bcs)

    def compute_J(snes, x, J_mat, P_mat):
        J_mat.zeroEntries()
        fem.petsc.assemble_matrix(J_mat, fem.form(J), bcs=bcs)
        J_mat.assemble()

    snes.setFunction(compute_F, b)
    snes.setJacobian(compute_J, A)
    snes.solve(None, u.x.petsc_vec)
    u.x.scatter_forward()

    return snes.getConvergedReason() > 0, snes.getIterationNumber()


class TestNewtonLSSolver:
    """Test Newton line search solver."""

    def test_solver_converges(self, nonlinear_poisson):
        """Verify Newton solver runs and converges."""
        domain, V, u, F, bcs = nonlinear_poisson

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        set_snes_options_newtonls(
            snes, options_prefix="newtonlsconv_", log_iterations=False
        )

        converged, iterations = solve_nonlinear(snes, u, F, bcs)

        assert converged
        # Magic number: expected iterations for this specific problem
        # If this changes, something in the solver or problem has changed
        assert iterations == 2, f"Expected 2 iterations, got {iterations}"

    def test_options_are_set(self, nonlinear_poisson):
        """Verify SNES options are set in PETSc database."""
        domain, V, u, F, bcs = nonlinear_poisson

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        prefix = "newtonlsset_"
        set_snes_options_newtonls(snes, options_prefix=prefix, log_iterations=False)

        # Check options are set in PETSc database
        opts = PETSc.Options()
        assert opts[f"{prefix}snes_type"] == "newtonls"
        assert opts[f"{prefix}snes_linesearch_type"] == "none"
        assert opts[f"{prefix}snes_rtol"] == "1e-06"


# ============================================================================
# SNESSetterRegistry Tests
# ============================================================================


class TestSNESSetterRegistry:
    """Test SNESSetterRegistry class."""

    def setup_method(self):
        """Save registry state before each test."""
        # Store original setters to restore after test
        self._original_setters = SNESSetterRegistry._setters.copy()

    def teardown_method(self):
        """Restore registry state after each test."""
        SNESSetterRegistry._setters.clear()
        SNESSetterRegistry._setters.update(self._original_setters)

    def test_register_and_retrieve_setter(self):
        """Test registering and retrieving a SNES setter."""

        # Create a simple test setter
        def test_setter(snes, **kwargs):
            pass

        # Register it
        SNESSetterRegistry.register("test_solver", test_setter)

        # Retrieve it
        retrieved = SNESSetterRegistry.get("test_solver")
        assert retrieved is test_setter

    def test_default_setters_registered(self):
        """Test that default setters are registered."""
        # Should have at least 'default' and 'newtonls'
        setters = SNESSetterRegistry.available_setters()
        assert "default" in setters
        assert "newtonls" in setters

    def test_get_default_setter(self):
        """Test retrieving the default SNES setter."""
        setter = SNESSetterRegistry.get("default")
        assert setter is set_snes_options_newtonls

    def test_get_newtonls_setter(self):
        """Test retrieving the newtonls SNES setter."""
        setter = SNESSetterRegistry.get("newtonls")
        assert setter is set_snes_options_newtonls

    def test_get_nonexistent_setter_raises_error(self):
        """Test that requesting nonexistent setter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown SNES setter 'nonexistent'"):
            SNESSetterRegistry.get("nonexistent")

    def test_available_setters_returns_list(self):
        """Test that available_setters returns a list of strings."""
        setters = SNESSetterRegistry.available_setters()
        assert isinstance(setters, list)
        assert len(setters) >= 2
        assert all(isinstance(name, str) for name in setters)

    def test_register_overwrites_existing(self):
        """Test that registering with same name overwrites previous."""

        def setter1(snes, **kwargs):
            pass

        def setter2(snes, **kwargs):
            pass

        SNESSetterRegistry.register("overwrite_test", setter1)
        assert SNESSetterRegistry.get("overwrite_test") is setter1

        SNESSetterRegistry.register("overwrite_test", setter2)
        assert SNESSetterRegistry.get("overwrite_test") is setter2

    def test_setter_execution_via_registry(self, nonlinear_poisson):
        """Test that a setter retrieved from registry works correctly."""
        domain, V, u, F, bcs = nonlinear_poisson

        snes = PETSc.SNES().create(MPI.COMM_WORLD)

        # Get setter from registry
        setter = SNESSetterRegistry.get("default")

        # Execute it
        setter(snes, options_prefix="registry_", log_iterations=False)

        # Verify options were set
        opts = PETSc.Options()
        assert opts["registry_snes_type"] == "newtonls"

    def test_registry_singleton_behavior(self):
        """Test that registry maintains singleton behavior across calls."""

        # Register a setter
        def custom_setter(snes, **kwargs):
            pass

        SNESSetterRegistry.register("singleton_test", custom_setter)

        # Access registry again - should still have the registered setter
        assert "singleton_test" in SNESSetterRegistry.available_setters()
        assert SNESSetterRegistry.get("singleton_test") is custom_setter

    def test_error_message_lists_available_setters(self):
        """Test that error message lists available setters."""
        with pytest.raises(ValueError) as exc_info:
            SNESSetterRegistry.get("invalid_name")

        error_msg = str(exc_info.value)
        assert "Available:" in error_msg
        # Should list some of the available setters
        available = SNESSetterRegistry.available_setters()
        for setter_name in available[:2]:  # Check at least first 2
            assert setter_name in error_msg
