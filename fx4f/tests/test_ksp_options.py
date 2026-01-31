"""
Lightweight tests for KSP solver options.

Focus:
- Code coverage: every line runs
- Consistency: iteration counts remain stable
- Options verification: all PETSc options are used
"""

import pytest
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from basix.ufl import element
import ufl

from fx4f.solvers.ksp_options import (
    KSPSetterRegistry,
    set_ksp_options_direct,
    set_ksp_options_gmres_boomeramg,
    set_ksp_options_cg_jacobi,
    set_ksp_options_cg_boomeramg,
)
from fx4f.solvers.ksp_options_stokes import (
    set_ksp_options_stokes,
    set_ksp_options_schur_gmres_boomeramg_jacobi,
)


@pytest.fixture
def simple_poisson():
    """Create minimal Poisson problem with BCs: -Δu = f."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = fem.functionspace(domain, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Weak form: ∫ ∇u·∇v dx = ∫ f·v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    L = f * v * ufl.dx

    # Add Dirichlet BC to make problem well-posed
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

    # Assemble system
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    A = assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()

    b = assemble_vector(fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    return A, b, V, domain


class TestDirectSolver:
    """Test direct solver - focus on coverage and option usage."""

    def test_runs_without_error(self, simple_poisson):
        """Verify direct solver runs successfully."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_direct(
            ksp, options_prefix="directnoerror_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Solver completed
        assert ksp.getIterationNumber() >= 1

    def test_options_are_set(self, simple_poisson):
        """Verify all options are set in PETSc.Options()."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "directset_"
        set_ksp_options_direct(ksp, options_prefix=prefix, log_iterations=False)

        # Check options were set in PETSc database
        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "preonly"
        assert opts[f"{prefix}pc_type"] == "lu"


class TestGMRESHypre:

    class TestUserOptions:
        """Test that user options are correctly forwarded to PETSc.Options."""

        def test_user_options_are_set(self, simple_poisson):
            """Verify that user-supplied options are set in PETSc.Options."""
            A, b, V, domain = simple_poisson

            ksp = PETSc.KSP().create(MPI.COMM_WORLD)
            ksp.setOperators(A)

            prefix = "gmreshypreuser_"
            user_options = {
                "ksp_rtol": "1e-12",
                "ksp_atol": "1e-14",
                "pc_hypre_boomeramg_max_iter": "2",
            }
            set_ksp_options_gmres_boomeramg(
                ksp, options_prefix=prefix, log_iterations=False, options=user_options
            )

            opts = PETSc.Options()
            for key, val in user_options.items():
                assert (
                    opts[f"{prefix}{key}"] == val
                ), f"Option {prefix}{key} not set correctly"

    """Test GMRES+Hypre solver."""

    def test_runs_without_error(self, simple_poisson):
        """Verify GMRES solver runs successfully."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_gmres_boomeramg(
            ksp, options_prefix="gmreshyprenoerror_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Check solver ran some iterations
        assert ksp.getIterationNumber() > 0

    def test_options_are_set(self, simple_poisson):
        """Verify GMRES options are set in database."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "gmreshypreset_"
        set_ksp_options_gmres_boomeramg(
            ksp, options_prefix=prefix, log_iterations=False
        )

        # Check options were set in PETSc database
        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "gmres"
        assert opts[f"{prefix}pc_type"] == "hypre"
        assert opts[f"{prefix}pc_hypre_type"] == "boomeramg"

    def test_direct_vs_iterative_solution(self, simple_poisson):
        """Verify direct and iterative solvers give same solution."""
        A, b, V, domain = simple_poisson

        # Direct solution
        ksp_direct = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_direct.setOperators(A)
        set_ksp_options_direct(
            ksp_direct, options_prefix="gmreshyprecmpdirect_", log_iterations=False
        )
        x_direct = A.createVecRight()
        ksp_direct.solve(b, x_direct)

        # Iterative solution
        ksp_iter = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_iter.setOperators(A)
        set_ksp_options_gmres_boomeramg(
            ksp_iter, options_prefix="gmreshyprecmpiter_", log_iterations=False
        )
        x_iter = A.createVecRight()
        ksp_iter.solve(b, x_iter)

        # Solutions should match (within tolerance)
        x_diff = x_direct.copy()
        x_diff.axpy(-1.0, x_iter)
        diff_norm = x_diff.norm()

        assert diff_norm < 1e-6, f"Solutions differ by {diff_norm}"

    def test_iteration_counts_stable(self, simple_poisson):
        """Verify iteration count matches expected value (magic number)."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_gmres_boomeramg(
            ksp, options_prefix="gmreshyprestable_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Magic number: expected iterations for this specific problem
        # If this changes, something in the solver or problem has changed
        assert (
            ksp.getIterationNumber() == 4
        ), f"Expected 4 iterations, got {ksp.getIterationNumber()}"


class TestSetterRegistry:
    """Test KSPSetterRegistry singleton."""

    def test_all_setters_callable(self):
        """Verify all registered setters are callable."""
        for name in KSPSetterRegistry.available_setters():
            setter = KSPSetterRegistry.get(name)
            assert callable(setter), f"{name} is not callable"

    def test_default_exists(self):
        """Verify default setter is registered."""
        assert "default" in KSPSetterRegistry.available_setters()

    def test_get_method(self):
        """Verify .get() method works."""
        setter = KSPSetterRegistry.get("default")
        assert callable(setter)

    def test_stokes_schur_registered(self):
        """Verify stokes_schur is registered from ksp_options_stokes."""
        assert "stokes_schur" in KSPSetterRegistry.available_setters()
        assert KSPSetterRegistry.get("stokes_schur") == set_ksp_options_stokes

    def test_schur_selfp_registered(self):
        """Verify schur_selfp is registered from ksp_options_stokes."""
        assert "schur_selfp" in KSPSetterRegistry.available_setters()
        assert (
            KSPSetterRegistry.get("schur_selfp")
            == set_ksp_options_schur_gmres_boomeramg_jacobi
        )


class TestCGJacobi:
    """Test CG solver with Jacobi preconditioning."""

    def test_runs_without_error(self, simple_poisson):
        """Verify CG-Jacobi solver runs successfully."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_cg_jacobi(
            ksp, options_prefix="cgjacnoerror_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Check solver converged
        assert ksp.getConvergedReason() > 0

    def test_options_are_set(self, simple_poisson):
        """Verify CG options are set in database."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "cgjacset_"
        set_ksp_options_cg_jacobi(ksp, options_prefix=prefix, log_iterations=False)

        # Check options were set in PETSc database
        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "cg"
        assert opts[f"{prefix}pc_type"] == "jacobi"

    def test_iteration_counts_stable(self, simple_poisson):
        """Verify iteration count matches expected value (magic number)."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_cg_jacobi(
            ksp, options_prefix="cgjacstable_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Magic number: expected iterations for this specific problem
        # If this changes, something in the solver or problem has changed
        assert (
            ksp.getIterationNumber() == 9
        ), f"Expected 9 iterations, got {ksp.getIterationNumber()}"


class TestCGBoomerAMG:
    """Test CG solver with BoomerAMG preconditioning."""

    def test_runs_without_error(self, simple_poisson):
        """Verify CG-BoomerAMG solver runs successfully."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_cg_boomeramg(
            ksp, options_prefix="cgamgnoerror_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Check solver converged
        assert ksp.getConvergedReason() > 0

    def test_options_are_set(self, simple_poisson):
        """Verify CG and hypre options are set in database."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "cgamgset_"
        set_ksp_options_cg_boomeramg(ksp, options_prefix=prefix, log_iterations=False)

        # Check options were set in PETSc database
        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "cg"
        assert opts[f"{prefix}pc_type"] == "hypre"
        assert opts[f"{prefix}pc_hypre_type"] == "boomeramg"

    def test_direct_vs_cg_solution(self, simple_poisson):
        """Verify direct and CG solvers give same solution."""
        A, b, V, domain = simple_poisson

        # Direct solution
        ksp_direct = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_direct.setOperators(A)
        set_ksp_options_direct(
            ksp_direct, options_prefix="cgamgcmpdirect_", log_iterations=False
        )
        x_direct = A.createVecRight()
        ksp_direct.solve(b, x_direct)

        # CG solution
        ksp_cg = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_cg.setOperators(A)
        set_ksp_options_cg_boomeramg(
            ksp_cg, options_prefix="cgamgcmpiter_", log_iterations=False
        )
        x_cg = A.createVecRight()
        ksp_cg.solve(b, x_cg)

        # Compare solutions
        x_diff = x_direct.copy()
        x_diff.axpy(-1.0, x_cg)
        diff_norm = x_diff.norm()

        assert diff_norm < 1e-6, f"Solution difference too large: {diff_norm}"

    def test_iteration_counts_stable(self, simple_poisson):
        """Verify iteration count matches expected value (magic number)."""
        A, b, V, domain = simple_poisson

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_cg_boomeramg(
            ksp, options_prefix="cgamgstable_", log_iterations=False
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Magic number: expected iterations for this specific problem
        # If this changes, something in the solver or problem has changed
        assert (
            ksp.getIterationNumber() == 3
        ), f"Expected 3 iterations, got {ksp.getIterationNumber()}"
