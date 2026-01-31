from treelog4dolfinx import log

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import ufl

from petsc4py import PETSc
from numpy.typing import NDArray
import dolfinx

# Import utility functions from ksp_options
from .ksp_options import (
    get_fieldsplit_is,
    set_pc_hypre_boomeramg,
    ksp_monitor,
    _apply_user_options,
)

__all__ = [
    "set_ksp_options_stokes",
    "set_ksp_options_schur_gmres_boomeramg_jacobi",
    "set_approximate_schur_inverse",
    "SchurInvApprox",
    "set_nullspace",
]


def set_ksp_options_schur_gmres_boomeramg_jacobi(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for the Krylov solver (linear solver). Sets to
    `gmres` iterative solver with upper Schur-based preconditioning. AMG is
    used for preconditioning of A00. The Schur complement is explicitly formed
    as A11-A10inv(diag(A00))A01, and its action is approximated using Jacobi.

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace, optional
        Solver context, required only when system matrix is not nested.
        If matrix is not nested, must contain:
        - W : dolfinx.fem.FunctionSpace
            Function space of mixed elements.
        - WV_map : NDArray, optional
            A00 subspace dofmap. If None, computed from W.sub(0).collapse().
        - WQ_map : NDArray, optional
            A11 subspace dofmap. If None, computed from W.sub(1).collapse().
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set a nullspace as a constant in A11, by default False
    log_iterations : bool, optional
        Whether to log solver iterations, by default True
    """
    A, _ = ksp.getOperators()
    if nullspace:
        set_nullspace(A)

    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "gmres"
    opts[f"{prefix}ksp_pc_side"] = (
        "right"  # Use right preconditioning: ( K M^-1 ) M x = b. This preserves the residual.
    )

    pc = ksp.getPC()  # Preconditioning object
    pc.setType("fieldsplit")  # Different preconditioning for different blocks

    is_u, is_p = get_fieldsplit_is(A=A, solver_context=solver_context)
    pc.setFieldSplitIS(("u", is_u), ("p", is_p))
    ksp_u, ksp_p = pc.getFieldSplitSubKSP()
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.SELFP)

    prefix_u = prefix + "u_"
    ksp_u.setOptionsPrefix(prefix_u)
    opts[f"{prefix_u}ksp_type"] = "preonly"  # Apply preconditioner as-if matrix solve
    set_pc_hypre_boomeramg(opts, prefix_u)

    prefix_p = prefix + "p_"
    ksp_p.setOptionsPrefix(prefix_p)
    opts[f"{prefix_p}ksp_type"] = "preonly"  # Apply preconditioner as-if matrix solve
    opts[f"{prefix_p}pc_type"] = "jacobi"  # Diagonal inverse

    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp_u.setFromOptions()
    ksp_p.setFromOptions()
    ksp.setFromOptions()

    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_ksp_options_stokes(
    ksp: PETSc.KSP,
    solver_context=None,
    options_prefix: str = "",
    options: dict[str, str] | None = None,
    nullspace: bool = False,
    log_iterations: bool = True,
    **kwargs,
) -> None:
    """
    PETSc option setter for Stokes problems using MINRES with custom Schur
    approximation. The approximate Schur inverse is specified by parameters
    in the solver_context. Options are the pressure Mass matrix (Mp), the
    pressure stiffness matrix (Kp), or their addition (MpKp).

    Parameters
    ----------
    ksp : PETSc.KSP
        Krylov subspace solver object.
    solver_context : SimpleNamespace
        Solver context with required attributes:
        - Q : dolfinx.fem.FunctionSpace
            Pressure function space for Schur inverse approximation.
        - nu : float
            Kinematic viscosity.
        - dt_inv : float
            Inverse of time step (1/dt). May be zero.
        - pctype : str
            Preconditioner type: "Mp", "Kp", or "MpKp".

        Additionally required if matrix is not nested:
        - W : dolfinx.fem.FunctionSpace
            Mixed function space for velocity and pressure.
        - WV_map : NDArray, optional
            Velocity subspace dofmap. If None, computed from W.sub(0).collapse().
        - WQ_map : NDArray, optional
            Pressure subspace dofmap. If None, computed from W.sub(1).collapse().
    options_prefix : str, optional
        Additional prefix to append to solver's existing prefix, by default ""
    options : dict[str, str], optional
        Additional user options, by default None
    nullspace : bool, optional
        Whether to set pressure nullspace, by default True
    log_iterations : bool, optional
        Whether to log solver iterations (monitor), by default True
    """
    A, _ = ksp.getOperators()
    if nullspace:
        set_nullspace(A)

    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if prefix is None:
        prefix = ""
    prefix = prefix + options_prefix
    ksp.setOptionsPrefix(prefix)

    opts[f"{prefix}ksp_type"] = "minres"
    opts[f"{prefix}ksp_error_if_not_converged"] = True
    # opts[f"{prefix}ksp_view"] = None # For testing
    ksp.setTolerances(rtol=1e-9)

    pc = ksp.getPC()  # Preconditioning object
    pc.setType("fieldsplit")  # Different preconditioning for different blocks
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    is_u, is_p = get_fieldsplit_is(A=A, solver_context=solver_context)
    pc.setFieldSplitIS(("u", is_u), ("p", is_p))
    ksp_u, ksp_p = pc.getFieldSplitSubKSP()

    # Approximate schur inverse
    set_approximate_schur_inverse(solver_context)

    prefix_u = prefix + "u_"
    ksp_u.setOptionsPrefix(prefix_u)
    opts[f"{prefix_u}ksp_type"] = "preonly"
    set_pc_hypre_boomeramg(opts, prefix_u)

    prefix_p = prefix + "p_"
    ksp_p.setOptionsPrefix(prefix_p)
    opts[f"{prefix_p}ksp_type"] = "preonly"
    opts[f"{prefix_p}pc_type"] = "python"
    opts[f"{prefix_p}pc_python_type"] = __name__ + ".SchurInvApprox"
    if solver_context.pctype in ["a", "c"]:
        opts[f"{prefix_p}SchurInvMp_ksp_type"] = "preonly"
        opts[f"{prefix_p}SchurInvMp_pc_type"] = "bjacobi"
    if solver_context.pctype in ["b", "c"]:
        opts[f"{prefix_p}SchurInvMp_ksp_type"] = "preonly"
        opts[f"{prefix_p}SchurInvKp_pc_type"] = "gamg"

    _apply_user_options(opts, options or {}, prefix)

    # Set the options
    ksp.setFromOptions()
    ksp_u.setFromOptions()
    ksp_p.setFromOptions()
    if log_iterations:
        ksp.setMonitor(ksp_monitor)


def set_approximate_schur_inverse(solver_context):
    sc = solver_context

    # Approximate schur inverse
    p = ufl.TrialFunction(sc.Q)
    q = ufl.TestFunction(sc.Q)

    if sc.pctype in ["Mp", "MpKp"]:
        Mp = assemble_matrix(fem.form(1 / sc.nu * p * q * ufl.dx))  # For dt_inv << nu
        Mp.assemble()
        Mp.setOption(PETSc.Mat.Option.SPD, True)
        SchurInvApprox.Mp = Mp
    if sc.pctype in ["Kp", "MpKp"]:
        Kp = assemble_matrix(
            fem.form(1 / sc.dt_inv * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx)
        )  # For nu << dt_inv
        Kp.assemble()
        Kp.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        SchurInvApprox.Kp = Kp


class SchurInvApprox:
    Mp = None
    Kp = None

    def setUp(self, pc):
        self.Mp_ksp = PETSc.KSP().create(comm=pc.comm)
        self.Mp_ksp.setOptionsPrefix(pc.getOptionsPrefix() + "SchurInvMp_")
        self.Mp_ksp.setOperators(self.Mp)
        self.Mp_ksp.setFromOptions()
        self.Kp_ksp = PETSc.KSP().create(comm=pc.comm)
        self.Kp_ksp.setOptionsPrefix(pc.getOptionsPrefix() + "SchurInvKp_")
        self.Kp_ksp.setOperators(self.Kp)
        self.Kp_ksp.setFromOptions()

    def apply(self, pc, x, y):
        if self.Mp is not None and self.Kp is not None:
            (z,) = self.get_work_vecs(x, 1)
        else:
            z = y

        if self.Kp is not None:
            self.Kp_ksp.solve(x, z)  # z = K_p^{-1} x
        if self.Mp is not None:
            self.Mp_ksp.solve(x, y)  # y = M_p^{-1} x
            if self.Kp is not None:
                y.axpy(1.0, z)  # y = y + z

    def get_work_vecs(self, v: PETSc.Vec, N: int) -> tuple[PETSc.Vec]:
        try:
            vecs = self._work_vecs
        except AttributeError:
            self._work_vecs = vecs = tuple(v.duplicate() for i in range(N))
        return vecs


def set_nullspace(A):
    """Set pressure nullspace for enclosed Stokes problem."""
    # For nested matrices, we need to manually create a compatible nested vector
    # since createVecRight() doesn't always return nested vectors
    if A.getType() == "nest":
        # Get the index sets to determine the structure
        nested_IS = A.getNestISs()

        # Create individual vectors for each block
        mat_nest = A.getNestSubMatrix(0, 0)
        vec0 = mat_nest.createVecRight() if mat_nest else A.createVecRight()

        mat_nest = A.getNestSubMatrix(1, 1) or A.getNestSubMatrix(1, 0)
        vec1 = mat_nest.createVecLeft() if mat_nest else A.createVecLeft()

        # Create nested vector
        null_vec = PETSc.Vec().createNest([vec0, vec1])
        n0, n1 = null_vec.getNestSubVecs()
        n0.set(0.0)
        n1.set(1.0)
    else:
        # For non-nested matrices, this won't work properly
        raise NotImplementedError("set_nullspace only works with nested matrices")

    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A), "Nullspace test failed"
    A.setNullSpace(nsp)


# Register Stokes-specific setters with the KSP registry
from .ksp_options import KSPSetterRegistry

KSPSetterRegistry.register("stokes_schur", set_ksp_options_stokes)
KSPSetterRegistry.register("schur_selfp", set_ksp_options_schur_gmres_boomeramg_jacobi)
