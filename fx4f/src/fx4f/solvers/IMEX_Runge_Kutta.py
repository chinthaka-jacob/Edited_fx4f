"""
Implicit-Explicit (IMEX) Runge-Kutta solver for time-dependent problems.

This module provides a general-purpose IMEX RK solver that decouples implicit
and explicit terms. It manages stage solutions, preconditioner setup, and
RK tableau definitions for various schemes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.la.petsc import _zero_vector
from treelog4dolfinx import LogTime

__all__ = ["IMEXRKSolver", "RKTableau", "RKTableauRegistry"]


class IMEXRKSolver(ABC):
    """
    Abstract Implicit-Explicit Runge-Kutta solver for coupled implicit-explicit systems.

    This class implements an IMEX Runge-Kutta time integration scheme for problems of the form:

        du/dt = f_imp(t, u) + f_exp(t, u)

    where f_imp is the implicit (stiff) term treated with an implicit RK method, and
    f_exp is the explicit (non-stiff) term treated with an explicit RK method. Both terms
    are advanced using synchronized RK tableaux with identical stage times.

    The solver provides a template for:
    - Multi-stage RK computations with automatic stage caching
    - RHS assembly with selective acceleration caching (only stages needed in future stages)
    - Linear system solving via PETSc KSP (Krylov subspace methods)
    - Boundary condition enforcement at each stage

    Subclasses must implement the abstract methods to define problem-specific physics:
    - assemble_base_acceleration: Time-stepping contribution (typically dt_inv * u_prev)
    - assemble_implicit_acceleration: Implicit RHS (e.g., viscous diffusion)
    - assemble_explicit_acceleration: Explicit RHS (e.g., advection, pressure gradient)
    - assemble_implicit_matrix: Implicit system matrix for stage solves
    - apply_bcs_to_rhs: Boundary condition application to RHS vectors

    Workflow:
    1. Subclass provides RK tableaux via __init__
    2. User calls initialize() to assemble stage KSP solvers
    3. User calls solve_timestep(t, dt) repeatedly to advance in time
    4. Each timestep internally computes stages 1 to stages-1, updating stage_solution_vec

    Attributes
    ----------
    RK_I : RKTableau
        Implicit Runge-Kutta tableau (Butcher array for implicit stages)
    RK_E : RKTableau
        Explicit Runge-Kutta tableau (Butcher array for explicit stages)
    stages : int
        Number of RK stages, derived from tableaux
    t : float
        Current simulation time (updated after each timestep)
    dt : float
        Timestep size; can be set during initialization or in solve_timestep() calls
    stage_solution_vec : PETSc.Vec
        Current stage solution vector (nested for block systems); updated during timesteps
    rhs_vec : PETSc.Vec
        Cached RHS vector reused across stages to minimize allocations
    f_base : PETSc.Vec
        Cached base acceleration vector (time derivative contribution)
    f_imp_stages : list[PETSc.Vec or None]
        Cached implicit acceleration vectors for each stage (None if not needed in future)
    f_exp_stages : list[PETSc.Vec or None]
        Cached explicit acceleration vectors for each stage (None if not needed in future)
    stage_ksps : dict
        Cached PETSc KSP solvers, keyed by effective timestep scaling (c_i * a_ii)
        to avoid re-creating solvers for repeated stage timesteps

    Notes
    -----
    - The solver validates that implicit and explicit tableaux have matching stages,
      stage times (c), and correct first row ([0]) and final row structure.
    - Nested (block) vectors are fully supported; ghost updates are automatically
      applied for distributed parallel computation.
    - Stage accelerations are cached only if needed in subsequent stages, reducing memory
      and computational overhead.
    - Linear solvers (KSP) are cached per unique effective timestep to avoid redundant setup.

    Examples
    --------
    Subclass implementation for a simple advection-diffusion problem::

        class AdvectionDiffusionSolver(IMEXRKSolver):
            def assemble_base_acceleration(self, b, stage_t, stage_dt, stage_solution_vec):
                # Assemble (u_prev - u) / dt contribution into vector b
                pass

            def assemble_implicit_acceleration(self, b, stage_t, stage_dt, stage_solution_vec):
                # Assemble viscous diffusion term into vector b
                pass

            def assemble_explicit_acceleration(self, b, stage_t, stage_dt, stage_solution_vec):
                # Assemble advection term into vector b
                pass

            def assemble_implicit_matrix(self, stage_dt_effective):
                # Assembles (I + dt*stage_dt_effective*nu*Laplacian)
                pass

            def apply_bcs_to_rhs(self, rhs_vec, stage_t, stage_effective_dt):
                # Apply Dirichlet/Neumann boundary conditions to rhs_vec in-place
                pass

    See Also
    --------
    RKTableau : Butcher tableaux representation for RK methods
    RKTableauRegistry : Registry for predefined IMEX schemes
    """

    def __init__(
        self,
        IMEX_schemes: list[RKTableau, RKTableau],
        t: float = 0,
        dt: float = None,
        stage_solution_vec: PETSc.Vec = None,
    ):
        """
        Initialize IMEX RK solver.

        Parameters
        ----------
        IMEX_schemes : tuple[RKTableau, RKTableau]
            Tuple of (implicit tableau, explicit tableau). Can be retrieved
            from RKTableauRegistry.get(name)
        t : float, optional
            Initial time (default: 0)
        dt : float, optional
            Timestep size (default: None, must be set before solve_timestep)
        stage_solution_vec : PETSc.Vec
            Stage solution vector (nested for block systems)

        Raises
        ------
        ValueError
            If tableaux are inconsistent (different stages, c values, or invalid structure)
        """
        # Tableaux
        self.RK_I, self.RK_E = IMEX_schemes
        self._validate_tableaux()
        self.stages = self.RK_I.stages

        # External attributes
        self.stage_solution_vec = stage_solution_vec
        self.dt = dt
        self.t = t

        # Cached vectors
        self.rhs_vec = stage_solution_vec.duplicate()
        self.f_base = stage_solution_vec.duplicate()
        self.f_imp_stages = [None] * self.stages
        self.f_exp_stages = [None] * self.stages
        for stage in range(self.stages):
            if (
                sum(abs(a[stage]) for a in self.RK_I.a[stage:]) > 0
            ):  # This stage's implicit acceleration is needed in future stages
                self.f_imp_stages[stage] = stage_solution_vec.duplicate()
            if (
                sum(abs(a[stage]) for a in self.RK_E.a[stage:]) > 0
            ):  # This stage's explicit acceleration is needed in future stages
                self.f_exp_stages[stage] = stage_solution_vec.duplicate()

        # Cached solvers
        self.stage_ksps = {}

    def _validate_tableaux(self):
        """Validate tableau consistency."""
        if self.RK_I.stages != self.RK_E.stages:
            raise ValueError(
                "Implicit and explicit schemes must have the same number of stages"
            )
        if self.RK_I.c != self.RK_E.c:
            raise ValueError(
                "Implicit and explicit schemes must have the same time-step fractions"
            )
        if self.RK_I.a[0] != [0]:
            raise ValueError("First row of implicit tableau must be [0]")
        if self.RK_E.a[0] != [0]:
            raise ValueError("First row of explicit tableau must be [0]")
        if self.RK_I.b != self.RK_I.a[-1]:
            raise ValueError("Implicit final coefficients must equal last stage row")

    def initialize(self):
        """Assemble stage solvers. Must be called after subclass is fully initialized."""
        self._cache_stage_solvers()

    # =====================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =====================================================================

    @abstractmethod
    def assemble_base_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float,
        stage_dt: float,
        stage_solution_vec: PETSc.Vec,
    ) -> None:
        """
        Assemble base acceleration vector (time derivative term) for a given stage.

        This represents the time-stepping contribution (typically dt_inv * u_prev).

        Parameters
        ----------
        b : PETSc.Vec
            Output vector to assemble into (nested for block systems)
        stage_t : float
            Time at current stage
        stage_dt : float
            Scaled timestep for stage (dt * c_i)
        stage_solution_vec : PETSc.Vec
            Solution vector at current stage
        """
        pass

    @abstractmethod
    def assemble_implicit_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float,
        stage_dt: float,
        stage_solution_vec: PETSc.Vec,
    ) -> None:
        """
        Assemble implicit acceleration vector for a given stage.

        This represents the implicit part of the system (e.g., diffusion term).

        Parameters
        ----------
        b : PETSc.Vec
            Output vector to assemble into (nested for block systems)
        stage_t : float
            Time at current stage
        stage_dt : float
            Scaled timestep for stage (dt * c_i)
        stage_solution_vec : PETSc.Vec
            Solution vector at current stage
        """
        pass

    @abstractmethod
    def assemble_explicit_acceleration(
        self,
        b: PETSc.Vec,
        stage_t: float,
        stage_dt: float,
        stage_solution_vec: PETSc.Vec,
    ) -> None:
        """
        Assemble explicit acceleration vector for a given stage.

        This represents the explicit part of the system (e.g., advection term).

        Parameters
        ----------
        b : PETSc.Vec
            Output vector to assemble into (nested for block systems)
        stage_t : float
            Time at current stage
        stage_dt : float
            Scaled timestep for stage (dt * c_i)
        stage_solution_vec : PETSc.Vec
            Solution vector at current stage
        """
        pass

    @abstractmethod
    def assemble_implicit_matrix(self, stage_dt_effective: float) -> PETSc.Mat:
        """
        Assemble implicit system matrix for a given stage.

        This is the matrix that defines the implicit linear system to solve
        at each stage (e.g., the Stokes system for Navier-Stokes).

        Parameters
        ----------
        stage_dt_effective : float
            Effective timestep scaling for stage (dt * c_i * a_ii)

        Returns
        -------
        PETSc.Mat
            Assembled implicit system matrix (nested for block systems)
        """
        pass

    @abstractmethod
    def apply_bcs_to_rhs(
        self, rhs_vec: PETSc.Vec, stage_t: float, stage_effective_dt: float
    ):
        """
        Apply boundary conditions to RHS vector.

        This is problem-specific and should be implemented by subclasses.

        Parameters
        ----------
        rhs_vec : PETSc.Vec
            RHS vector to modify in-place (nested for block systems)
        stage_t : float
            Time at current stage
        stage_effective_dt : float
            Effective scaled timestep (dt * c_i * a_ii)

        Notes
        -----
        Implementations should modify rhs_vec in-place and ensure proper
        ghost value updates for parallel computation.
        """
        for rhs_vec_sub in rhs_vec.getNestSubVecs():
            rhs_vec_sub.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
        pass

    def pre_stage_hook(self, stage_t: float, previous_stage_solution_vec: PETSc.Vec):
        """
        Hook called before each stage solve.

        Can be overridden by subclasses to perform actions before stage assembly.

        Parameters
        ----------
        stage_t : float
            Time at current stage
        previous_stage_solution_vec : PETSc.Vec
            Solution vector at previous stage
        """
        pass

    def post_stage_hook(self, stage_t: float, stage_solution_vec: PETSc.Vec):
        """
        Hook called after each stage solve.

        Can be overridden by subclasses to perform actions after stage assembly.

        Parameters
        ----------
        stage_t : float
            Time at current stage
        stage_solution_vec : PETSc.Vec
            Solution vector at current stage
        """
        pass

    def pre_solve_hook(self, t: float, previous_solution_vec: PETSc.Vec):
        """
        Hook called before the full timestep solve begins.

        Can be overridden by subclasses to perform actions before timestep assembly.

        Parameters
        ----------
        t : float
            Time at start of timestep
        previous_solution_vec : PETSc.Vec
            Solution vector at start of timestep
        """
        pass

    def post_solve_hook(self, t: float, solution_vec: PETSc.Vec):
        """
        Hook called after the full timestep solve is complete.

        Can be overridden by subclasses to perform actions after timestep assembly.

        Parameters
        ----------
        t : float
            Time at end of timestep
        solution_vec : PETSc.Vec
            Solution vector at end of timestep
        """
        pass

    # =====================================================================
    # Public API
    # =====================================================================

    def solve_timestep(
        self, t: float | None = None, dt: float | None = None, logtime: LogTime = None
    ):
        """
        Advance solution by one time step using IMEX RK scheme.

        Computes all stages sequentially, updating self.stage_solution_vec
        with the final solution and advancing self.t.

        Parameters
        ----------
        t : float, optional
            Time at start of timestep (if None, uses self.t)
        dt : float, optional
            Timestep size (if None, uses self.dt)
        logtime : LogTime, optional
            Logger object with log_time() method for timing information

        Raises
        ------
        RuntimeError
            If dt is not set and not provided as argument
        """
        # Store time information locally for access in acceleration assemblers
        self.t = t if t is not None else self.t
        self.dt = dt if dt is not None else self.dt
        if self.dt is None:
            raise RuntimeError(
                "Timestep 'dt' is not set; call solve_timestep(dt, t, ...), initialize with dt, or set self.dt before assembling accelerations"
            )

        self.pre_solve_hook(t=self.t, previous_solution_vec=self.stage_solution_vec)
        for stage in range(1, self.stages):
            self._cache_acceleration_vectors(stage - 1)
            self._pre_stage_hook(stage)
            self._compute_rhs(stage)
            self._solve_stage(stage)
            self._post_stage_hook(stage)
            if logtime is not None:
                logtime.log_time(f"Stage {stage} computed")

        self.t += self.dt
        self.post_solve_hook(t=self.t, solution_vec=self.stage_solution_vec)

    def _compute_rhs(self, stage: int) -> PETSc.Vec:
        """
        Assemble RHS for a given stage.

        Parameters
        ----------
        stage : int
            Stage number (1 to stages-1)

        Returns
        -------
        PETSc.Vec
            RHS vector with boundary conditions applied
        """
        rhs_vec = self.rhs_vec
        _zero_vector(rhs_vec)

        # Add base acceleration
        a_ss = self.RK_I.a[stage][stage]
        c = self.RK_I.c[stage]
        self._assemble_acceleration_vector(
            self.f_base, self.assemble_base_acceleration, c
        )
        rhs_vec.axpy(1 / a_ss, self.f_base)

        # Add contributions from prior stages
        for j, (a_imp, a_exp) in enumerate(
            zip(self.RK_I.a[stage][:-1], self.RK_E.a[stage][:-1])
        ):
            if a_imp != 0.0:
                f_imp_j = self.f_imp_stages[j]
                rhs_vec.axpy(-a_imp / a_ss, f_imp_j)
            if a_exp != 0.0:
                f_exp_j = self.f_exp_stages[j]
                rhs_vec.axpy(-a_exp / a_ss, f_exp_j)

        c = self.RK_I.c[stage]
        self.apply_bcs_to_rhs(
            rhs_vec, stage_t=self.t + self.dt * c, stage_effective_dt=self.dt * c * a_ss
        )
        return rhs_vec

    def _solve_stage(self, stage: int):
        """
        Solve the linear system for a given stage.

        Parameters
        ----------
        stage : int
            Stage number (1 to stages-1)
        """
        ksp_solver = self._get_stage_solver(stage)
        ksp_solver.solve(self.rhs_vec, x=self.stage_solution_vec)
        _scatter_petsc_vec_forward(self.stage_solution_vec)

    def _cache_acceleration_vectors(self, stage: int):
        """
        Pre-compute and cache implicit and explicit acceleration vectors of
        the `stage`. Assumes the `self.solution_vec` at the time of computation
        corresponds to the solution of `stage`.

        Parameters
        ----------
        stage : int
            Current stage number
        """
        # Check if this stage's accelerations are needed in future stages
        f_imp = self.f_imp_stages[stage]
        if f_imp is not None:
            c = self.RK_I.c[stage]
            self._assemble_acceleration_vector(
                f_imp, self.assemble_implicit_acceleration, c
            )

        f_exp = self.f_exp_stages[stage]
        if f_exp is not None:
            c = self.RK_E.c[stage]
            self._assemble_acceleration_vector(
                f_exp, self.assemble_explicit_acceleration, c
            )

    def _assemble_acceleration_vector(
        self, accel_vec: PETSc.Vec, assembly_func, c: float
    ) -> None:
        """
        Assemble and cache an acceleration vector for a stage.

        Generic method to assemble, zero, and scatter an acceleration vector.
        Assumes self.stage_solution_vec contains the solution at the current stage.

        Parameters
        ----------
        accel_vec : PETSc.Vec
            Acceleration vector to assemble into
        assembly_func : callable
            Function to call for assembly (e.g., assemble_implicit_acceleration)
        c : float
            Time-step fraction for the stage (c_i)
        """
        _zero_vector(accel_vec)
        assembly_func(
            b=accel_vec,
            stage_t=self.t + self.dt * c,
            stage_dt=self.dt * c,
            stage_solution_vec=self.stage_solution_vec,
        )
        _scatter_petsc_vec_backward(accel_vec)

    def _cache_stage_solvers(self):
        """
        Assemble and cache linear KSP solvers for each stage.

        Creates one KSP per unique effective timestep scaling (dt * c_i * a_ii).
        Called during initialize() to set up solvers before timestep advances.

        Raises
        ------
        RuntimeError
            If dt is not set
        """
        if self.dt is None:
            raise RuntimeError(
                "Timestep 'dt' is not set; call solve_timestep(dt, t, ...), initialize with dt, or set self.dt before assembling stage solvers"
            )
        for stage in range(1, self.stages):
            a, c = self.RK_I.a[stage][stage], self.RK_I.c[stage]
            if not c * a in self.stage_ksps.keys():
                A = self.assemble_implicit_matrix(self.dt * c * a)
                ksp = PETSc.KSP().create(MPI.COMM_WORLD)
                ksp.setOperators(A)
                self.stage_ksps[c * a] = ksp

    def _get_stage_solver(self, stage: int) -> PETSc.KSP:
        """
        Retrieve cached KSP solver for a given stage.

        Parameters
        ----------
        stage : int
            Stage number (1 to stages-1)

        Returns
        -------
        PETSc.KSP
            Cached Krylov solver object for this stage's effective timestep
        """
        a, c = self.RK_I.a[stage][stage], self.RK_I.c[stage]
        return self.stage_ksps[a * c]

    def _pre_stage_hook(self, stage: int):
        """
        Forward to pre_stage_hook with stage time.

        Parameters
        ----------
        stage : int
            Current stage number
        """
        c = self.RK_I.c[stage]
        stage_t = self.t + self.dt * c
        self.pre_stage_hook(
            stage_t=stage_t, previous_stage_solution_vec=self.stage_solution_vec
        )

    def _post_stage_hook(self, stage: int):
        """
        Forward to post_stage_hook with stage time.

        Parameters
        ----------
        stage : int
            Current stage number
        """
        c = self.RK_I.c[stage]
        stage_t = self.t + self.dt * c
        self.post_stage_hook(
            stage_t=stage_t, stage_solution_vec=self.stage_solution_vec
        )


class RKTableau:
    """
    Runge-Kutta tableau for implicit or explicit schemes.

    Attributes
    ----------
    c : list[float]
        Time-step fractions c_i for each stage (i.e., where forcing applies)
    a : list[list[float]]
        Butcher tableau coefficients a_ij (weightings for j-th acceleration in i-th stage)
    b : list[float]
        Final step coefficients b_j (weightings for j-th acceleration in final solution)
    stages : int
        Number of stages in the method
    """

    def __init__(self, c: list[float], a: list[list[float]], b: list[float]):
        """
        Initialize an RK tableau.

        Parameters
        ----------
        c : list[float]
            Time-step fractions
        a : list[list[float]]
            Butcher tableau coefficients
        b : list[float]
            Final step coefficients
        """
        self.c = c
        self.a = a
        self.b = b
        self._validate()

    @property
    def stages(self) -> int:
        """Number of stages in the method."""
        return len(self.c)

    def _validate(self):
        """Validate tableau consistency."""
        if len(self.a) != self.stages:
            raise ValueError(
                f"Tableau 'a' has {len(self.a)} rows, expected {self.stages}"
            )
        if len(self.b) != self.stages:
            raise ValueError(
                f"Tableau 'b' has {len(self.b)} elements, expected {self.stages}"
            )


class RKTableauRegistry:
    """Registry for IMEX RK schemes."""

    _schemes: dict = {}

    @classmethod
    def register(cls, name: str, implicit: "RKTableau", explicit: "RKTableau") -> None:
        """
        Register a new IMEX scheme.

        Parameters
        ----------
        name : str
            Name of the scheme (e.g., 'RK-TR')
        implicit : RKTableau
            Implicit RK tableau
        explicit : RKTableau
            Explicit RK tableau
        """
        if implicit.stages != explicit.stages:
            raise ValueError(
                "Implicit and explicit schemes must have the same number of stages"
            )
        if implicit.c != explicit.c:
            raise ValueError(
                "Implicit and explicit schemes must have the same time-step fractions"
            )
        cls._schemes[name] = (implicit, explicit)

    @classmethod
    def get(cls, name: str) -> tuple[RKTableau, RKTableau]:
        """
        Retrieve an IMEX scheme by name.

        Parameters
        ----------
        name : str
            Name of the scheme

        Returns
        -------
        tuple[RKTableau, RKTableau]
            Implicit and explicit tableaux

        Raises
        ------
        KeyError
            If scheme is not registered
        ValueError
            If scheme name is invalid
        """
        if name not in cls._schemes:
            available = ", ".join(cls._schemes.keys())
            raise ValueError(f"Unknown IMEX scheme '{name}'. Available: {available}")
        return cls._schemes[name]

    @classmethod
    def available_schemes(cls) -> list[str]:
        """Return list of available scheme names."""
        return list(cls._schemes.keys())


# Register default schemes
RKTableauRegistry.register(
    "RK-TR",
    implicit=RKTableau(c=[0, 1, 1], a=[[0], [0, 1], [0.5, 0, 0.5]], b=[0.5, 0, 0.5]),
    explicit=RKTableau(c=[0, 1, 1], a=[[0], [1, 0], [0.5, 0.5, 0]], b=[0.5, 0.5, 0]),
)


def _scatter_petsc_vec_backward(vec: PETSc.Vec):
    """
    Scatter PETSc vector backward (ghostUpdate with ADD and REVERSE).

    Handles both regular and nested vectors, accumulating ghost values.

    Parameters
    ----------
    vec : PETSc.Vec
        Vector to scatter (regular or nested for block systems)
    """
    if vec.getType() != PETSc.Vec.Type.NEST:
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    else:
        for sub_vec in vec.getNestSubVecs():
            sub_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )


def _scatter_petsc_vec_forward(vec: PETSc.Vec):
    """
    Scatter PETSc vector forward (ghostUpdate with INSERT and FORWARD).

    Handles both regular and nested vectors, distributing values to ghosts.

    Parameters
    ----------
    vec : PETSc.Vec
        Vector to scatter (regular or nested for block systems)
    """
    if vec.getType() != PETSc.Vec.Type.NEST:
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        for sub_vec in vec.getNestSubVecs():
            sub_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
