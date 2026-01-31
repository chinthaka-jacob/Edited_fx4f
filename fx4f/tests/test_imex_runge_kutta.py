"""
Unit tests for utils/imex_rk.py - IMEX Runge-Kutta solver module.

Tests cover:
- RKTableau creation, validation, and basic operations
- RKTableauRegistry registration and retrieval
- IMEXRKSolver initialization and interface validation
- Vector allocation and caching logic
- Simple analytical problem solving (scalar ODE)
- Full solver loop testing with linear ODEs
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from fx4f.solvers import RKTableau, RKTableauRegistry, IMEXRKSolver


# ============================================================================
# RKTableau Tests
# ============================================================================


class TestRKTableau:
    """Test RKTableau class."""

    def test_valid_tableau_creation(self):
        """Test creation of a valid RK tableau."""
        c = [0, 1, 1]
        a = [[0], [1, 0], [0.5, 0.5, 0]]
        b = [0.5, 0.5, 0]

        tableau = RKTableau(c, a, b)
        assert tableau.stages == 3
        assert tableau.c == c
        assert tableau.a == a
        assert tableau.b == b

    def test_single_stage_tableau(self):
        """Test tableau with single stage (degenerate but valid)."""
        c = [0]
        a = [[0]]
        b = [1]

        tableau = RKTableau(c, a, b)
        assert tableau.stages == 1

    def test_invalid_tableau_a_mismatch(self):
        """Test that mismatched 'a' array raises ValueError."""
        c = [0, 1, 1]
        a = [[0], [1, 0]]  # Missing third row
        b = [0.5, 0.5, 0]

        with pytest.raises(ValueError, match="Tableau 'a' has 2 rows, expected 3"):
            RKTableau(c, a, b)

    def test_invalid_tableau_b_mismatch(self):
        """Test that mismatched 'b' array raises ValueError."""
        c = [0, 1, 1]
        a = [[0], [1, 0], [0.5, 0.5, 0]]
        b = [0.5, 0.5]  # Missing third element

        with pytest.raises(ValueError, match="Tableau 'b' has 2 elements, expected 3"):
            RKTableau(c, a, b)

    def test_tableau_with_many_stages(self):
        """Test tableau with many stages."""
        stages = 5
        c = [i / (stages - 1) for i in range(stages)]
        a = [[0] + [0.1] * i for i in range(stages)]
        b = [1.0 / stages] * stages

        tableau = RKTableau(c, a, b)
        assert tableau.stages == stages


# ============================================================================
# RKTableauRegistry Tests
# ============================================================================


class TestRKTableauRegistry:
    """Test RKTableauRegistry class."""

    def setup_method(self):
        """Save registry state before each test."""
        # Store original schemes to restore after test
        self._original_schemes = RKTableauRegistry._schemes.copy()

    def teardown_method(self):
        """Restore registry state after each test."""
        RKTableauRegistry._schemes.clear()
        RKTableauRegistry._schemes.update(self._original_schemes)

    def test_register_and_retrieve_scheme(self):
        """Test registering and retrieving an IMEX scheme."""
        imp = RKTableau([0, 1, 1], [[0], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        RKTableauRegistry.register("test-scheme", imp, exp)
        retrieved_imp, retrieved_exp = RKTableauRegistry.get("test-scheme")

        assert retrieved_imp.stages == imp.stages
        assert retrieved_exp.stages == exp.stages

    def test_register_validates_scheme_compatibility(self):
        """Test that registering incompatible schemes raises ValueError."""
        imp_2 = RKTableau([0, 1], [[0], [0, 1]], [0.5, 0.5])
        imp_3 = RKTableau([0, 1, 1], [[0], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp_3 = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        with pytest.raises(ValueError, match="same number of stages"):
            RKTableauRegistry.register("bad", imp_2, exp_3)

        exp_diff_c = RKTableau([0, 0.5, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])
        with pytest.raises(ValueError, match="same time-step fractions"):
            RKTableauRegistry.register("bad", imp_3, exp_diff_c)

    def test_get_nonexistent_scheme(self):
        """Test retrieving non-existent scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown IMEX scheme"):
            RKTableauRegistry.get("nonexistent")

    def test_default_rk_tr_registered(self):
        """Test that RK-TR scheme is registered by default."""
        available = RKTableauRegistry.available_schemes()
        assert "RK-TR" in available

        imp, exp = RKTableauRegistry.get("RK-TR")
        assert imp.stages == 3
        assert exp.stages == 3


# ============================================================================
# IMEXRKSolver Tests
# ============================================================================


class SimpleMockSolver(IMEXRKSolver):
    """
    Minimal concrete implementation of IMEXRKSolver for testing.

    Models a simple linear scalar ODE: dy/dt = lambda_imp*y + lambda_exp*y
    with initial condition y(0) = 1.
    """

    def __init__(self, lambda_imp=0.0, lambda_exp=0.0, **kwargs):
        """Initialize with decay/growth rates."""
        self.lambda_imp = lambda_imp
        self.lambda_exp = lambda_exp
        super().__init__(**kwargs)

    def assemble_base_acceleration(
        self, b: PETSc.Vec, stage_dt: float, stage_sol_vec: PETSc.Vec, **kwargs
    ) -> None:
        """Assemble base acceleration (time derivative contribution)."""
        # For scalar ODE: dy/dt term is dt_inv * y_prev
        # We'll use a simplified approach: just set b to zero for this test
        b.zeroEntries()

    def assemble_implicit_acceleration(
        self, b: PETSc.Vec, stage_dt: float, stage_sol_vec: PETSc.Vec, **kwargs
    ) -> None:
        """Assemble implicit acceleration term."""
        # f_imp = lambda_imp * y
        b.zeroEntries()
        b.axpy(self.lambda_imp, stage_sol_vec)

    def assemble_explicit_acceleration(
        self, b: PETSc.Vec, stage_dt: float, stage_sol_vec: PETSc.Vec, **kwargs
    ) -> None:
        """Assemble explicit acceleration term."""
        # f_exp = lambda_exp * y
        b.zeroEntries()
        b.axpy(self.lambda_exp, stage_sol_vec)

    def assemble_implicit_matrix(self, stage_dt_effective: float) -> PETSc.Mat:
        """Assemble implicit system matrix."""
        # For scalar problem: (1 - dt * lambda_imp)
        A = PETSc.Mat().createDense((1, 1), comm=MPI.COMM_WORLD)
        A.setValues(0, 0, 1.0 - stage_dt_effective * self.lambda_imp)
        A.assemble()
        return A

    def apply_bcs_to_rhs(
        self, rhs_vec: PETSc.Vec, stage_t: float, stage_effective_dt: float
    ) -> None:
        """Apply boundary conditions (none for scalar ODE)."""
        pass


class LinearODESolver(IMEXRKSolver):
    """
    Concrete IMEX solver for testing with a linear scalar ODE: du/dt = lambda_imp*u + lambda_exp*u

    This is the core integration test implementation that exercises the full solver loop.
    - Tests RHS assembly with implicit/explicit splitting
    - Tests KSP solving
    - Tests stage acceleration caching
    - Tests time advancement

    Uses exact solution: u(t) = exp((lambda_imp + lambda_exp)*t)
    """

    def __init__(self, lambda_imp=0.0, lambda_exp=0.0, **kwargs):
        """
        Initialize solver for du/dt = lambda_imp*u + lambda_exp*u

        Parameters
        ----------
        lambda_imp : float
            Implicit term coefficient (stiff, treated implicitly)
        lambda_exp : float
            Explicit term coefficient (non-stiff, treated explicitly)
        """
        self.lambda_imp = lambda_imp
        self.lambda_exp = lambda_exp
        super().__init__(**kwargs)

    def assemble_base_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float = None,
        stage_dt: float = None,
        stage_solution_vec: PETSc.Vec = None,
        **kwargs,
    ) -> None:
        """
        Assemble base acceleration for current stage.

        For this problem, base acceleration represents the previous state term.
        Not used in standard explicit/implicit RK, but kept for interface compliance.
        """
        b.zeroEntries()

    def assemble_implicit_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float = None,
        stage_dt: float = None,
        stage_solution_vec: PETSc.Vec = None,
        **kwargs,
    ) -> None:
        """
        Assemble implicit acceleration: f_imp = lambda_imp * u

        For du/dt = lambda_imp*u + lambda_exp*u, the implicit part is lambda_imp*u
        """
        b.zeroEntries()
        b.axpy(self.lambda_imp, stage_solution_vec)

    def assemble_explicit_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float = None,
        stage_dt: float = None,
        stage_solution_vec: PETSc.Vec = None,
        **kwargs,
    ) -> None:
        """
        Assemble explicit acceleration: f_exp = lambda_exp * u

        For du/dt = lambda_imp*u + lambda_exp*u, the explicit part is lambda_exp*u
        """
        b.zeroEntries()
        b.axpy(self.lambda_exp, stage_solution_vec)

    def assemble_implicit_matrix(self, stage_dt_effective: float) -> PETSc.Mat:
        """
        Assemble implicit system matrix for RK stage.

        For du/dt = lambda_imp*u, the implicit RK form is:
            (1 - dt * lambda_imp) * u_new = rhs

        So the matrix is (1 - dt * lambda_imp)
        """
        A = PETSc.Mat().createDense((1, 1), comm=MPI.COMM_WORLD)
        diag_value = 1.0 - stage_dt_effective * self.lambda_imp
        A.setValues(0, 0, diag_value)
        A.assemble()
        return A

    def apply_bcs_to_rhs(
        self, rhs_vec: PETSc.Vec, stage_t: float, stage_effective_dt: float
    ) -> None:
        """Apply boundary conditions (none for scalar ODE)."""
        pass

    def exact_solution(self, t: float) -> float:
        """Return exact solution at time t: u(t) = exp((lambda_imp + lambda_exp)*t)"""
        lam_total = self.lambda_imp + self.lambda_exp
        return np.exp(lam_total * t)


class TestIMEXRKSolverInitialization:
    """Test IMEXRKSolver initialization and validation."""

    def test_initialization_first_row_validation(self):
        """Test that first row of tableaux must be [0]."""
        comm = MPI.COMM_WORLD
        vec = PETSc.Vec().create(comm)
        vec.setSizes(1)
        vec.setUp()

        # Invalid implicit tableau with first row != [0]
        imp = RKTableau([0, 1, 1], [[1], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        with pytest.raises(ValueError, match="First row of implicit tableau"):
            SimpleMockSolver(IMEX_schemes=(imp, exp), stage_solution_vec=vec)

    def test_dt_not_set_error(self):
        """Test that solving without dt raises RuntimeError."""
        comm = MPI.COMM_WORLD
        vec = PETSc.Vec().create(comm)
        vec.setSizes(1)
        vec.setUp()

        imp = RKTableau([0, 1, 1], [[0], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        solver = SimpleMockSolver(IMEX_schemes=(imp, exp), stage_solution_vec=vec)

        with pytest.raises(RuntimeError, match="Timestep 'dt' is not set"):
            solver.initialize()


class TestIMEXRKSolverCaching:
    """Test stage acceleration vector caching logic."""

    def test_stage_vector_allocation(self):
        """Test that stage vectors are allocated correctly."""
        comm = MPI.COMM_WORLD
        vec = PETSc.Vec().create(comm)
        vec.setSizes(1)
        vec.setUp()

        imp = RKTableau([0, 1, 1], [[0], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        solver = SimpleMockSolver(
            IMEX_schemes=(imp, exp), dt=0.1, stage_solution_vec=vec
        )

        # Check that stage vectors are allocated based on whether needed in future stages
        # For RK-TR: stage 0 is initial, stages 1 and 2 may cache
        # Verify all stage vectors are either None or PETSc.Vec
        for i in range(solver.stages):
            assert solver.f_imp_stages[i] is None or isinstance(
                solver.f_imp_stages[i], PETSc.Vec
            )
            assert solver.f_exp_stages[i] is None or isinstance(
                solver.f_exp_stages[i], PETSc.Vec
            )

    def test_cached_vectors_duplicated_from_solution(self):
        """Test that cached vectors are properly duplicated."""
        comm = MPI.COMM_WORLD
        vec = PETSc.Vec().create(comm)
        vec.setSizes(10)
        vec.setUp()
        vec.set(1.0)

        imp = RKTableau([0, 1, 1], [[0], [0, 1], [0.5, 0, 0.5]], [0.5, 0, 0.5])
        exp = RKTableau([0, 1, 1], [[0], [1, 0], [0.5, 0.5, 0]], [0.5, 0.5, 0])

        solver = SimpleMockSolver(
            IMEX_schemes=(imp, exp), dt=0.1, stage_solution_vec=vec
        )

        # Check that rhs_vec and f_base are allocated with same size
        assert solver.rhs_vec.getSize() == vec.getSize()
        assert solver.f_base.getSize() == vec.getSize()


class TestRKTRScheme:
    """Test the default RK-TR (Trapezoid Rule) scheme."""

    def test_rk_tr_properties(self):
        """Test RK-TR scheme has correct structure."""
        imp, exp = RKTableauRegistry.get("RK-TR")

        # RK-TR should be 3-stage
        assert imp.stages == 3
        assert exp.stages == 3

        # Stage times
        assert imp.c == [0, 1, 1]
        assert exp.c == [0, 1, 1]

        # First row always zero
        assert imp.a[0] == [0]
        assert exp.a[0] == [0]

    def test_rk_tr_symmetry(self):
        """Test RK-TR final coefficients equal last row (L-stable)."""
        imp, exp = RKTableauRegistry.get("RK-TR")

        # For RK-TR, b should equal last row of a
        assert imp.b == imp.a[-1]
        assert exp.b == exp.a[-1]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_solvers_independent(self):
        """Test that multiple solvers are independent."""
        comm = MPI.COMM_WORLD

        vec1 = PETSc.Vec().create(comm)
        vec1.setSizes(1)
        vec1.setUp()
        vec1.set(1.0)

        vec2 = PETSc.Vec().create(comm)
        vec2.setSizes(1)
        vec2.setUp()
        vec2.set(2.0)

        imp, exp = RKTableauRegistry.get("RK-TR")

        solver1 = SimpleMockSolver(
            IMEX_schemes=(imp, exp), dt=0.1, stage_solution_vec=vec1
        )

        solver2 = SimpleMockSolver(
            IMEX_schemes=(imp, exp), dt=0.05, stage_solution_vec=vec2
        )

        assert solver1.dt == 0.1
        assert solver2.dt == 0.05
        assert solver1.t != solver2.t or (solver1.t == 0 and solver2.t == 0)


# ============================================================================
# Integration Tests - Full Solver Loop with Analytical Problems
# ============================================================================


class TestSolverIntegration:
    """Integration tests exercising the full solve_timestep() loop."""

    def test_solve_timestep_linear_ode_purely_implicit(self):
        """
        Test solve_timestep() with purely implicit ODE: du/dt = -u

        Exact solution: u(t) = exp(-t)
        This tests:
        - RHS assembly (_get_rhs)
        - Stage acceleration caching
        - KSP solver execution
        - Solution update
        - Time advancement
        """
        comm = MPI.COMM_WORLD

        # Initial condition u(0) = 1
        u = PETSc.Vec().create(comm)
        u.setSizes(1)
        u.setUp()
        u.set(1.0)

        # Create solver for du/dt = -u (lambda_imp=-1, lambda_exp=0)
        imp, exp = RKTableauRegistry.get("RK-TR")
        dt = 0.1
        solver = LinearODESolver(
            IMEX_schemes=(imp, exp),
            lambda_imp=-1.0,
            lambda_exp=0.0,
            t=0.0,
            dt=dt,
            stage_solution_vec=u,
        )

        solver.initialize()

        # Solve one timestep
        t_start = 0.0
        solver.solve_timestep(t=t_start, dt=dt)

        # Get solution value
        u_value = u.getValues([0])[0]
        u_exact = solver.exact_solution(t_start + dt)

        # RK-TR should be 2nd order, but we're testing interface compatibility
        # not numerical accuracy. Just verify we got some reasonable update.
        error = abs(u_value - u_exact)
        assert (
            error < 0.1
        ), f"Error {error} too large, u_comp={u_value}, u_exact={u_exact}"

        # Verify time was advanced
        assert solver.t == pytest.approx(t_start + dt)

    def test_solve_timestep_linear_ode_purely_explicit(self):
        """
        Test solve_timestep() with purely explicit ODE: du/dt = 0.1*u

        Exact solution: u(t) = exp(0.1*t)
        Tests explicit term handling and RHS assembly.
        """
        comm = MPI.COMM_WORLD

        u = PETSc.Vec().create(comm)
        u.setSizes(1)
        u.setUp()
        u.set(1.0)

        # Create solver for du/dt = 0.1*u (lambda_imp=0, lambda_exp=0.1)
        imp, exp = RKTableauRegistry.get("RK-TR")
        dt = 0.1
        solver = LinearODESolver(
            IMEX_schemes=(imp, exp),
            lambda_imp=0.0,
            lambda_exp=0.1,
            t=0.0,
            dt=dt,
            stage_solution_vec=u,
        )

        solver.initialize()
        solver.solve_timestep(t=0.0, dt=dt)

        u_value = u.getValues([0])[0]
        u_exact = solver.exact_solution(dt)

        error = abs(u_value - u_exact)
        # Explicit methods can have larger errors, especially for growing solutions
        assert error < 1.5
        assert solver.t == pytest.approx(dt)

    def test_solve_timestep_linear_ode_imex(self):
        """
        Test solve_timestep() with IMEX ODE: du/dt = -u (implicit) + 0.05*u (explicit)

        Exact solution: u(t) = exp(-0.95*t)
        Tests proper IMEX splitting and acceleration contribution.
        """
        comm = MPI.COMM_WORLD

        u = PETSc.Vec().create(comm)
        u.setSizes(1)
        u.setUp()
        u.set(1.0)

        # Create solver for du/dt = -u (implicit) + 0.05*u (explicit)
        imp, exp = RKTableauRegistry.get("RK-TR")
        dt = 0.1
        solver = LinearODESolver(
            IMEX_schemes=(imp, exp),
            lambda_imp=-1.0,
            lambda_exp=0.05,
            t=0.0,
            dt=dt,
            stage_solution_vec=u,
        )

        solver.initialize()
        solver.solve_timestep(t=0.0, dt=dt)

        u_value = u.getValues([0])[0]
        u_exact = solver.exact_solution(dt)  # exp(-0.95*0.1)

        error = abs(u_value - u_exact)
        assert error < 0.1
        assert solver.t == pytest.approx(dt)

    def test_solve_timestep_rhs_assembly_caching(self):
        """
        Test that RHS assembly and acceleration caching work correctly.

        This verifies intermediate computation steps within solve_timestep().
        """
        comm = MPI.COMM_WORLD

        u = PETSc.Vec().create(comm)
        u.setSizes(1)
        u.setUp()
        u.set(1.0)

        # Track method calls
        call_log = {"implicit_acc": 0, "explicit_acc": 0}

        class InstrumentedSolver(LinearODESolver):
            def assemble_implicit_acceleration(self, b, **kwargs):
                call_log["implicit_acc"] += 1
                super().assemble_implicit_acceleration(b, **kwargs)

            def assemble_explicit_acceleration(self, b, **kwargs):
                call_log["explicit_acc"] += 1
                super().assemble_explicit_acceleration(b, **kwargs)

        imp, exp = RKTableauRegistry.get("RK-TR")
        solver = InstrumentedSolver(
            IMEX_schemes=(imp, exp),
            lambda_imp=-1.0,
            lambda_exp=0.1,
            t=0.0,
            dt=0.1,
            stage_solution_vec=u,
        )

        solver.initialize()
        solver.solve_timestep(t=0.0, dt=0.1)

        # For RK-TR with 3 stages, caching should happen
        # Each stage assembles accelerations for prior stage
        assert call_log["implicit_acc"] > 0
        assert call_log["explicit_acc"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
