"""
Lightweight tests for Stokes-specific KSP solver options.

Tests the fieldsplit and stokes_schur solvers on Stokes problems.
"""

import pytest
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from basix.ufl import element, mixed_element
import ufl
from ufl import div, grad, inner, sym, dx

from fx4f.solvers.ksp_options import set_ksp_options_direct
from fx4f.solvers.ksp_options_stokes import (
    set_ksp_options_schur_gmres_boomeramg_jacobi,
    set_ksp_options_stokes,
)


@pytest.fixture
def stokes_mixed_problem():
    """Create a simple 2D lid-driven cavity Stokes problem."""
    from types import SimpleNamespace

    # Create mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [(0, 0), (1, 1)], [8, 8], mesh.CellType.triangle
    )

    # Function spaces - create mixed space directly
    Ve = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
    Qe = element("Lagrange", domain.basix_cell(), 1)
    Ue = mixed_element([Ve, Qe])
    W = fem.functionspace(domain, Ue)

    # Get submaps
    _, WV_map = W.sub(0).collapse()
    Q, WQ_map = W.sub(1).collapse()

    # Boundary conditions
    def walls(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], 1)

    def lid(x):
        return np.isclose(x[1], 1)

    facets_wall = mesh.locate_entities_boundary(domain, domain.geometry.dim - 1, walls)
    dofs_walls = fem.locate_dofs_topological(
        (W.sub(0), W.sub(0).collapse()[0]), domain.geometry.dim - 1, facets_wall
    )
    bc_zero_u = fem.dirichletbc(
        fem.Function(W.sub(0).collapse()[0]), dofs_walls, W.sub(0)
    )

    facets_lid = mesh.locate_entities_boundary(domain, domain.geometry.dim - 1, lid)
    dofs_lid = fem.locate_dofs_topological(
        (W.sub(0), W.sub(0).collapse()[0]), domain.geometry.dim - 1, facets_lid
    )
    u_unit = fem.Function(W.sub(0).collapse()[0])
    u_unit.interpolate(lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1])))
    bc_unit_u = fem.dirichletbc(u_unit, dofs_lid, W.sub(0))

    bcs = [bc_zero_u, bc_unit_u]

    # Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Strain operator
    def epsilon(u):
        return sym(grad(u))

    # Stokes weak form
    nu = fem.Constant(domain, PETSc.ScalarType(1.0))
    a = inner(2 * nu * epsilon(u), epsilon(v)) * dx - div(v) * p * dx - q * div(u) * dx
    L = (
        inner(fem.Constant(domain, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0))), v)
        * dx
    )

    # Assemble into matrix
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    A = assemble_matrix(fem.form(a), bcs=bcs)
    A.assemble()

    b = assemble_vector(fem.form(L))
    fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Create solver context
    solver_context = SimpleNamespace(
        W=W, WV_map=WV_map, WQ_map=WQ_map, Q=Q, nu=nu, dt_inv=0, pctype="Mp"
    )

    return A, b, solver_context


class TestFieldsplitSchur:
    """Test fieldsplit Schur preconditioner for Stokes."""

    def test_solver_runs(self, stokes_mixed_problem):
        """Verify fieldsplit Schur solver runs successfully."""
        A, b, solver_context = stokes_mixed_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_schur_gmres_boomeramg_jacobi(
            ksp,
            solver_context=solver_context,
            options_prefix="stokesfieldsplitruns_",
            log_iterations=False,
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        assert ksp.getConvergedReason() > 0
        assert ksp.getIterationNumber() > 0

    def test_options_are_set(self, stokes_mixed_problem):
        """Verify fieldsplit options are set in PETSc database."""
        A, b, solver_context = stokes_mixed_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "stokesfieldsplitset_"
        set_ksp_options_schur_gmres_boomeramg_jacobi(
            ksp,
            solver_context=solver_context,
            options_prefix=prefix,
            log_iterations=False,
        )

        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "gmres"
        assert opts[f"{prefix}ksp_pc_side"] == "right"
        # pc_type is set programmatically, not via options
        assert ksp.getPC().getType() == "fieldsplit"

    def test_iteration_count_stable(self, stokes_mixed_problem):
        """Verify iteration count matches expected value (magic number)."""
        A, b, solver_context = stokes_mixed_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_schur_gmres_boomeramg_jacobi(
            ksp,
            solver_context=solver_context,
            options_prefix="stokesfieldsplitstable_",
            log_iterations=False,
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        # Magic number: expected iterations for this specific problem
        # If this changes, something in the solver or problem has changed
        assert (
            ksp.getIterationNumber() == 13
        ), f"Expected 11 iterations, got {ksp.getIterationNumber()}"

    def test_iterative_vs_direct_solution(self, stokes_mixed_problem):
        """Verify iterative and direct solvers give same solution.

        Note: Stokes problems have pressure nullspace (pressure defined up to constant).
        This test will be enabled once proper nullspace handling is implemented.
        """
        PETSc.Options().clear()
        A, b, solver_context_mixed = stokes_mixed_problem

        # Direct solution
        ksp_direct = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_direct.setOperators(A)
        set_ksp_options_direct(
            ksp_direct,
            options_prefix="stokesfieldsplitcmpdirect_",
            log_iterations=False,
        )
        x_direct = A.createVecRight()
        ksp_direct.solve(b, x_direct)

        # Iterative solution with fieldsplit
        ksp_iter = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_iter.setOperators(A)

        # Create extended solver context with all required fields
        set_ksp_options_schur_gmres_boomeramg_jacobi(
            ksp_iter,
            solver_context=solver_context_mixed,
            options_prefix="stokesfieldsplitcmpdirectiter_",
            log_iterations=False,
            options={"ksp_rtol": "1e-10", "ksp_atol": "1e-12"},
        )
        x_iter = A.createVecRight()
        ksp_iter.solve(b, x_iter)

        # Check that the iterative solver actually converged
        assert (
            ksp_iter.getConvergedReason() > 0
        ), f"Iterative solver did not converge, reason: {ksp_iter.getConvergedReason()}"

        # Solutions should match (within tolerance)
        x_diff = x_direct.copy()
        x_diff.axpy(-1.0, x_iter)
        diff_norm = x_diff.norm()

        assert diff_norm < 1e-6, f"Solutions differ by {diff_norm}"


@pytest.fixture
def stokes_nested_problem():
    """Create a Stokes problem with nested matrix structure for set_ksp_options_stokes."""
    from types import SimpleNamespace
    from dolfinx.la.petsc import create_vector_wrap

    # Create mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [(0, 0), (1, 1)], [8, 8], mesh.CellType.triangle
    )

    # Function spaces - separate V and Q
    Ve = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
    Qe = element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, Ve)
    Q = fem.functionspace(domain, Qe)

    # Boundary conditions
    def walls(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0)

    def lid(x):
        return np.isclose(x[1], 1)

    facets_wall = mesh.locate_entities_boundary(domain, domain.geometry.dim - 1, walls)
    dofs_walls = fem.locate_dofs_topological(V, domain.geometry.dim - 1, facets_wall)
    bc_zero_u = fem.dirichletbc(fem.Function(V), dofs_walls)

    facets_lid = mesh.locate_entities_boundary(domain, domain.geometry.dim - 1, lid)
    dofs_lid = fem.locate_dofs_topological(V, domain.geometry.dim - 1, facets_lid)
    u_unit = fem.Function(V)
    u_unit.interpolate(lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1])))
    bc_unit_u = fem.dirichletbc(u_unit, dofs_lid)

    bcs = [bc_zero_u, bc_unit_u]

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    # Constants
    nu = 1.0
    dt = 0.1
    nu_ = fem.Constant(domain, PETSc.ScalarType(nu))
    dt_inv_ = fem.Constant(domain, PETSc.ScalarType(1.0 / dt))

    # Strain operator
    def epsilon(u):
        return sym(grad(u))

    # Stokes weak form as nested blocks
    B00 = inner(dt_inv_ * u, v) * dx + inner(2 * nu_ * epsilon(u), epsilon(v)) * dx
    B01 = -div(v) * p * dx
    B10 = -q * div(u) * dx
    B11 = None
    B = [[fem.form(B00), fem.form(B01)], [fem.form(B10), B11]]

    # Assemble nested matrix
    from dolfinx.fem.petsc import assemble_matrix

    A = assemble_matrix(B, bcs=bcs, kind="nest")
    A.assemble()

    # Set matrix properties
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.getNestSubMatrix(0, 0).setOption(PETSc.Mat.Option.SPD, True)

    # Assemble RHS
    L0 = (
        inner(fem.Constant(domain, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0))), v)
        * dx
    )
    L1 = ufl.ZeroBaseForm((q,))
    L = [fem.form(L0), fem.form(L1)]

    from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
    from dolfinx.fem import bcs_by_block, extract_function_spaces

    b = assemble_vector(L, kind="nest")

    # Apply lifting
    bcs1 = bcs_by_block(extract_function_spaces(B, 1), bcs)
    apply_lifting(b, B, bcs=bcs1)

    # Ghost update
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set BC
    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
    set_bc(b, bcs0)

    # Create solver context
    solver_context = SimpleNamespace(Q=Q, nu=nu, dt_inv=1.0 / dt, pctype="Mp")

    return A, b, solver_context


class TestStokesSchur:
    """Test stokes_schur solver (MINRES with custom Schur approximation)."""

    def test_solver_runs(self, stokes_nested_problem):
        """Verify stokes_schur solver runs successfully with expected iteration count.

        Magic number test: tracks regression in solver performance.
        If iterations change, solver configuration or problem may have changed.
        """
        A, b, solver_context = stokes_nested_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_stokes(
            ksp,
            solver_context=solver_context,
            nullspace=True,
            monitor=False,
            options_prefix="stokesschurrun_",
        )

        x = b.duplicate()
        ksp.solve(b, x)

        assert ksp.getConvergedReason() > 0
        # Magic number: expected iterations for this specific problem
        assert (
            ksp.getIterationNumber() == 43
        ), f"Expected 43 iterations, got {ksp.getIterationNumber()}"

    def test_solution_matches_direct(self, stokes_nested_problem):
        """Verify iterative stokes_schur solver produces same result as direct solve.

        For Stokes problems, velocity should match exactly, but pressure can differ
        by an arbitrary constant due to the pressure nullspace.
        """
        A, b, solver_context = stokes_nested_problem

        # Direct solution
        ksp_direct = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_direct.setOperators(A.copy())
        set_ksp_options_direct(
            ksp_direct,
            nullspace=True,
            options_prefix="stokesschurdirect_",
            log_iterations=False,
        )
        x_direct = b.duplicate()
        ksp_direct.solve(b, x_direct)

        # Iterative solution with stokes_schur
        ksp_iter = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_iter.setOperators(A)
        set_ksp_options_stokes(
            ksp_iter,
            solver_context=solver_context,
            nullspace=True,
            monitor=False,
            options_prefix="stokesschuriter_",
        )
        x_iter = b.duplicate()
        ksp_iter.solve(b, x_iter)

        u_direct, p_direct = x_direct.getNestSubVecs()
        u_iter, p_iter = x_iter.getNestSubVecs()

        # Velocity should match exactly
        u_diff = u_direct.copy()
        u_diff.axpy(-1.0, u_iter)
        u_diff_norm = u_diff.norm()

        # Pressure can differ by a constant - check that difference is constant
        p_diff = p_direct.copy()
        p_diff.axpy(-1.0, p_iter)

        # Get pressure difference values
        p_diff_array = p_diff.getArray()

        # Check if variance of pressure difference is small (i.e., it's approximately constant)
        p_diff_mean = np.mean(p_diff_array)
        p_diff_variance = np.var(p_diff_array)

        assert u_diff_norm < 1e-6, f"Velocity solutions differ by {u_diff_norm}"
        assert (
            p_diff_variance < 1e-10
        ), f"Pressure difference is not constant (variance={p_diff_variance}, mean={p_diff_mean})"

    def test_options_are_set(self, stokes_nested_problem):
        """Verify stokes_schur options are set correctly."""
        A, b, solver_context = stokes_nested_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        prefix = "stokesschurset_"
        set_ksp_options_stokes(
            ksp,
            solver_context=solver_context,
            nullspace=False,
            monitor=False,
            options_prefix=prefix,
        )

        opts = PETSc.Options()
        assert opts[f"{prefix}ksp_type"] == "minres"
        assert ksp.getPC().getType() == "fieldsplit"

    def test_registry_contains_stokes_schur(self):
        """Verify stokes_schur is registered in KSPSetterRegistry."""
        from fx4f.solvers.ksp_options import KSPSetterRegistry

        available = KSPSetterRegistry.available_setters()
        assert "stokes_schur" in available

        setter = KSPSetterRegistry.get("stokes_schur")
        assert setter == set_ksp_options_stokes

    def test_solver_context_structure(self, stokes_nested_problem):
        """Verify solver context has required attributes."""
        A, b, solver_context = stokes_nested_problem

        # Check solver context has all required attributes
        assert hasattr(solver_context, "Q")
        assert hasattr(solver_context, "nu")
        assert hasattr(solver_context, "dt_inv")
        assert hasattr(solver_context, "pctype")

        # Verify they have correct types/values
        assert solver_context.nu == 1.0
        assert solver_context.dt_inv == 10.0
        assert solver_context.pctype == "Mp"


class TestCrossCompatibility:
    """Test that solvers work with both nested and mixed matrix structures."""

    def test_schur_selfp_with_nested_matrix(self, stokes_nested_problem):
        """Verify set_ksp_options_schur_gmres_boomeramg_jacobi works with nested matrix."""
        A, b, solver_context = stokes_nested_problem

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        # This solver doesn't need solver_context for nested matrices
        set_ksp_options_schur_gmres_boomeramg_jacobi(
            ksp,
            solver_context=None,
            options_prefix="stokecompatfieldsplitnested_",
            log_iterations=False,
        )

        x = b.duplicate()
        ksp.solve(b, x)

        assert ksp.getConvergedReason() > 0
        # Magic number: expected iterations for nested matrix with this solver
        assert (
            ksp.getIterationNumber() == 11
        ), f"Expected 11 iterations, got {ksp.getIterationNumber()}"

    def test_stokes_schur_with_mixed_matrix(self, stokes_mixed_problem):
        """Verify set_ksp_options_stokes works with mixed element matrix."""
        from types import SimpleNamespace

        A, b, solver_context_mixed = stokes_mixed_problem

        # Need to add Q, nu, dt_inv, pctype to the mixed solver_context
        # Extract Q from W
        W = solver_context_mixed.W
        Q = W.sub(1).collapse()[0]

        # Create extended solver context with all required fields
        solver_context = SimpleNamespace(
            W=W,
            WV_map=solver_context_mixed.WV_map,
            WQ_map=solver_context_mixed.WQ_map,
            Q=Q,
            nu=1.0,
            dt_inv=10.0,
            pctype="Mp",
        )

        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)

        set_ksp_options_stokes(
            ksp,
            solver_context=solver_context,
            nullspace=False,
            log_iterations=False,
            options_prefix="stokescompatschurmixed_",
        )

        x = A.createVecRight()
        ksp.solve(b, x)

        assert ksp.getConvergedReason() > 0
        # Magic number: expected iterations for mixed matrix with this solver
        assert (
            ksp.getIterationNumber() == 51
        ), f"Expected 42 iterations, got {ksp.getIterationNumber()}"
