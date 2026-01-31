from sympy import lambdify, diff, simplify
from sympy.abc import x, y, z, t
import numpy as np
from dolfinx import mesh, fem
import itertools

from numpy.typing import NDArray
from typing import Callable, Any
from pathlib import Path

# Optional dependency - load_mesh is only needed for mesh_from_checkpoint
try:
    from ..io import load_mesh
except ImportError:
    # Not available when module is used standalone or in certain test scenarios
    load_mesh = None

__all__ = [
    "ReferenceSolution",
    "AnalyticalSolution",
    "AnalyticalIncompressibleNavierStokesSolution",
]


class ReferenceSolution:
    """
    Base class for defining reference solutions, with functionality for
    setting up the problem (mesh, material parameters, BCs, periodicity,
    initial condition, etc), and access to reference data (quantities of
    interest).

    Design Philosophy
    -----------------
    This class provides a unified interface for accessing:

    1. **Fields** - Solution fields (velocity, pressure, etc.) that can be
       evaluated at spatial coordinates and optionally time
    2. **Facet/Element/DOF Getters** - Methods to identify mesh entities for
       applying boundary conditions, material properties, or constraints
    3. **Reference Data** - Benchmark values, DNS data, or other reference
       quantities for validation

    All access is through a registration pattern where child classes register
    their capabilities, and users query what's available via properties before
    accessing data via getter methods.

    Inheritance Guide
    -----------------
    Child classes should:

    1. **Call super().__init__()** first in their __init__ method

    2. **Register fields** using `register_field(name, callable)`:
       - The callable should return a function that takes coordinates (and
         optionally time) and returns field values
       - Example: `self.register_field("u_inflow", self._u_inflow)`

    3. **Populate getter dictionaries** in __init__:
       - `self._facets_getters` - Maps tags to functions returning facet indices
       - `self._elements_getters` - Maps tags to functions returning element indices
       - `self._dofs_getters` - Maps tags to functions returning DOF indices
       - Example: `self._facets_getters = {"inlet": self._get_inlet_facets}`

    4. **Register reference data** using `register_reference_data(key, value_or_getter)`:
       - For static values: `self.register_reference_data("Cd_max", 2.95)`
       - For computed values: `self.register_reference_data("timeseries", self._get_timeseries)`
       - Getters can accept parameters via **kwargs in get_reference_data()

    5. **Implement create_mesh()** to create and store self.domain

    6. **Optionally implement** periodic_boundary_indicator() and periodic_relation()
       for periodic boundary conditions

    User Interface
    --------------
    Users interact with reference solutions through:

    - **Discovery**: Check `available_fields`, `available_facet_getters`,
      `available_element_getters`, `available_dof_getters`, `available_reference_data`
      properties to see what's available

    - **Field Access**: `get_field(name, **kwargs)` returns a callable that
      evaluates the field at coordinates

    - **Mesh Entities**: `get_facets(tag)`, `get_elements(tag)`, `get_dofs(tag, V)`
      return entity indices for the given tag

    - **Reference Data**: `get_reference_data(key, **params)` retrieves benchmark
      or validation data

    - **Interpolation**: `interpolate_field(function, field, **kwargs)` directly
      interpolates a field onto a DOLFINx function

    Example
    -------
    >>> class MyFlow(ReferenceSolution):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.register_field("u", self._velocity)
    ...         self._facets_getters = {"inlet": self._get_inlet}
    ...         self.register_reference_data("Re", 100)
    ...
    ...     def _velocity(self):
    ...         def field(x):
    ...             return np.array([x[1], 0.0])
    ...         return field
    ...
    ...     def _get_inlet(self):
    ...         return mesh.locate_entities_boundary(...)
    ...
    ...     def create_mesh(self):
    ...         self.domain = mesh.create_rectangle(...)
    ...         return self.domain
    >>>
    >>> flow = MyFlow()
    >>> print(flow.available_fields)  # ['u']
    >>> u_field = flow.get_field("u")
    >>> values = u_field(coordinates)
    >>> inlet_facets = flow.get_facets("inlet")
    >>> Re = flow.get_reference_data("Re")
    """

    def __init__(self) -> None:
        self._facets_getters: dict[str, Callable] = {}
        self._elements_getters: dict[str, Callable] = {}
        self._dofs_getters: dict[str, Callable] = {}
        self._fields: dict[str, Callable] = {}
        self._reference_data: dict[str, Any] = {}
        self._reference_data_getters: dict[str, Callable] = {}
        self.domain: mesh.Mesh | None = None

    @property
    def available_fields(self) -> list[str]:
        """Read-only view of registered field names."""
        return list(self._fields.keys())

    @property
    def available_facet_getters(self) -> list[str]:
        """Read-only access to available facet getter tags."""
        return list(self._facets_getters.keys())

    @property
    def available_element_getters(self) -> list[str]:
        """Read-only access to available element getter tags."""
        return list(self._elements_getters.keys())

    @property
    def available_dof_getters(self) -> list[str]:
        """Read-only access to available DOF getter tags."""
        return list(self._dofs_getters.keys())

    @property
    def available_reference_data(self) -> list[str]:
        """Read-only access to available reference data keys."""
        all_keys = set(self._reference_data.keys()) | set(
            self._reference_data_getters.keys()
        )
        return sorted(list(all_keys))

    @property
    def dim(self):
        if hasattr(self, "_dim"):
            return self._dim
        if self.domain is not None:
            return self.domain.geometry.dim
        raise AttributeError(
            "Dimension is not set; initialize with a dimension or create a mesh first."
        )

    def mesh_from_checkpoint(self, filename: str | Path) -> mesh.Mesh:
        """
        Loads mesh from checkpoint, and stores it in object for interfacing
        with facet getters, etc.

        Parameters
        ----------
        filename : str | Path
            Path to file/directory, e.g., outputdir+"/solution.bp"

        Returns
        -------
        mesh.Mesh
            The stored mesh.

        Raises
        ------
        ImportError
            If load_mesh is not available (io module not imported).
        """
        if load_mesh is None:
            raise ImportError(
                "load_mesh is not available. "
                "Ensure the io module is properly installed and accessible."
            )
        self.domain: mesh.Mesh = load_mesh(filename)
        return self.domain

    def create_mesh(self) -> mesh.Mesh:
        """
        Creates the mesh for this particular reference problem, and stores
        it in object for interfacing with facet getters, etc.

        Returns
        -------
        mesh.Mesh
            The stored mesh.

        Raises
        ------
        NotImplementedError
            Must be overloaded.
        """
        raise NotImplementedError

    def get_facets(self, tag: str) -> NDArray:
        """
        For the particular boundary tag, obtains the marked mesh entities
        for, e.g., prescribing boundary conditions.

        Parameters
        ----------
        tag : str
            Tag, options differ per reference case implementation.

        Returns
        -------
        NDArray
            Indices (local to the process) of marked mesh entities.
        """
        return self._facets_getters[tag]()

    def get_elements(self, tag: str) -> NDArray:
        """
        For the particular element region tag, obtains the marked mesh entities
        for, e.g., assigning material properties or volume integrals.

        Parameters
        ----------
        tag : str
            Tag, options differ per reference case implementation.

        Returns
        -------
        NDArray
            Indices (local to the process) of marked mesh entities.
        """
        return self._elements_getters[tag]()

    def get_dofs(
        self,
        dof_region: str,
        V: fem.FunctionSpace | tuple[fem.FunctionSpace, fem.FunctionSpace],
    ) -> NDArray:
        """
        For the particular DOF region tag, obtains the marked mesh entities
        for, e.g., prescribing point constraints or DOF fixing.

        Parameters
        ----------
        dof_region : str
            Tag, options differ per reference case implementation.

        Returns
        -------
        NDArray
            Indices (local to the process) of marked mesh entities.
        """
        return self._dofs_getters[dof_region](V)

    def register_reference_data(
        self, key: str, value_or_getter: Any | Callable
    ) -> None:
        """
        Register reference data, either as a static value or a callable getter.

        Parameters
        ----------
        key : str
            Identifier for the reference data.
        value_or_getter : Any | Callable
            Either a direct value (e.g., dict, float) or a callable that
            generates the data when called with parameters.
        """
        if callable(value_or_getter):
            self._reference_data_getters[key] = value_or_getter
        else:
            self._reference_data[key] = value_or_getter

    def get_reference_data(self, key: str, **params) -> Any:
        """
        Retrieve reference data by key, optionally passing parameters for getters.

        Parameters
        ----------
        key : str
            Identifier for the reference data.
        **params
            Optional parameters to pass to getter functions.

        Returns
        -------
        Any
            The reference data value.

        Raises
        ------
        KeyError
            If the key is not found in reference data.
        """
        if key in self._reference_data:
            return self._reference_data[key]
        elif key in self._reference_data_getters:
            return self._reference_data_getters[key](**params)
        else:
            raise KeyError(
                f"Reference data key '{key}' not found. Available keys: {self.available_reference_data}"
            )

    def periodic_meshtag(self, tag: int, domain: mesh.Mesh = None) -> mesh.MeshTags:
        """
        Obtain the meshtag for defining a `multi_point_constraint` object,
        for enforcing periodicity. In case relevant for this reference
        case.

        Parameters
        ----------
        tag : int
            Tag.
        domain : mesh.Mesh, optional
            mesh (self.domain if unspecified), by default None

        Returns
        -------
        mesh.MeshTags
            A mesh tags object.
        """
        domain = self.domain if domain is None else domain
        facets = mesh.locate_entities_boundary(
            domain, self.dim - 1, self.periodic_boundary_indicator
        )
        arg_sort = np.argsort(facets)
        return mesh.meshtags(
            domain,
            self.dim - 1,
            facets[arg_sort],
            np.full(len(facets), tag, dtype=np.int32),
        )

    def periodic_boundary_indicator(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Marking function for indicating which mesh entities live on the to-
        be-mapped boundary.

        Parameters
        ----------
        x : NDArray[np.float64]
            Coordinates.

        Returns
        -------
        NDArray[np.bool_]
            Mask of mesh entities.

        Raises
        ------
        NotImplementedError
            Must be overloaded.
        """
        raise NotImplementedError

    def periodic_relation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Maps the given coordinates, that live on the to-be-mapped boundary
        to the opposite side.

        Parameters
        ----------
        x : NDArray[np.float64]
            Coordinates in.

        Returns
        -------
        NDArray[np.float64]
            Coordinates out.

        Raises
        ------
        NotImplementedError
            Must be overloaded.
        """
        raise NotImplementedError

    def interpolate_field(self, function: fem.Function, field: str, **kwargs) -> None:
        """
        Interpolates the `field` onto the provided `function`,
        with additional arguments passable as key word arguments (e.g.
        `t=0`).

        Parameters
        ----------
        function : fem.Function
            Onto which the field is interpolated
        field : str
            Field that must be interpolated.
        """
        function.interpolate(self.get_field(field, **kwargs))

    def get_field(self, field: str, **kwargs) -> Callable:
        """Access registered field and forward kwargs to its callable.

        Parameters
        ----------
        field : str
            Field name as registered.
        **kwargs
            Forwarded to the field's callable.

        Returns
        -------
        Callable
            The field callable with kwargs applied.
        """
        if field not in self._fields:
            raise KeyError(
                f"Field '{field}' not available. "
                f"Registered fields: {list(self._fields.keys())}"
            )
        return self._fields[field](**kwargs)

    def register_field(self, name: str, callable_field: Callable) -> None:
        """Register a field with a callable.

        Parameters
        ----------
        name : str
            Field name (e.g., "u_inflow", "p").
        callable_field : Callable
            Callable that produces the field values.
        """
        if not isinstance(name, str):
            raise TypeError(f"Field name must be a string, got {type(name)}")
        if not callable(callable_field):
            raise TypeError(f"Field value must be callable, got {type(callable_field)}")
        self._fields[name] = callable_field


class AnalyticalSolution(ReferenceSolution):
    """
    Base class for analytical solutions with symbolic math support.
    Extends ReferenceSolution with automatic lambdification of SymPy expressions
    and on-demand computation of time derivatives.

    Design Philosophy
    -----------------
    This class bridges symbolic mathematics (SymPy) and numerical computation
    (NumPy) by:

    1. **Lazy Lambdification** - SymPy expressions are converted to fast NumPy
       functions only when first accessed, with results cached for reuse

    2. **Automatic Differentiation** - Time derivatives are computed symbolically
       from base expressions and cached separately

    3. **Type-Aware Registration** - Fields are tagged as scalar/vector/tensor
       to handle their evaluation and stacking correctly

    Inheritance Guide
    -----------------
    Child classes should:

    1. **Call super().__init__(dim)** with dimension (2 or 3)
       - This sets up coordinate variables (x, y, t for 2D; x, y, z, t for 3D)

    2. **Define symbolic expressions** using SymPy:
       - Use self.vars_space for spatial variables [x, y] or [x, y, z]
       - Use t for time variable
       - Example: `u_expr = [sp.sin(x) * sp.cos(y) * sp.exp(-t), ...]`

    3. **Register fields with SymPy expressions**:
       - `self.register_field(name, expression, field_type)`
       - field_type: "scalar", "vector", or "tensor"
       - Example: `self.register_field("u", [ux_expr, uy_expr], "vector")`
       - Example: `self.register_field("p", p_expr, "scalar")`

    4. **Access time derivatives** via get_field():
       - Base field: `get_field("u", time_derivative=0)`
       - First derivative: `get_field("u", time_derivative=1)`
       - Second derivative: `get_field("u", time_derivative=2)`

    Internal Storage
    ----------------
    - `_expr_cache`: Maps field names to SymPy expressions
    - `_lambda_cache`: Maps (field, derivative_order) to lambdified functions
    - `_field_types`: Maps field names to "scalar"/"vector"/"tensor"
    - `_field_extra_args`: Tracks extra symbolic parameters per field

    Example
    -------
    >>> class MyAnalyticalFlow(AnalyticalSolution):
    ...     def __init__(self):
    ...         super().__init__(dim=2)
    ...         import sympy as sp
    ...         # Define symbolic expressions
    ...         ux = sp.sin(x) * sp.cos(y) * sp.exp(-t)
    ...         uy = -sp.cos(x) * sp.sin(y) * sp.exp(-t)
    ...         p = -0.25 * (sp.cos(2*x) + sp.cos(2*y)) * sp.exp(-2*t)
    ...
    ...         # Register fields
    ...         self.register_field("u", [ux, uy], "vector")
    ...         self.register_field("p", p, "scalar")
    >>>
    >>> flow = MyAnalyticalFlow()
    >>> coords = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> u_base = flow.get_field("u", time_derivative=0)(coords, t=1.0)
    >>> u_dot = flow.get_field("u", time_derivative=1)(coords, t=1.0)
    >>> p_vals = flow.get_field("p")(coords, t=1.0)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3 for analytical solutions")

        self._dim = dim

        # Coordinate variables derived from dimension
        if dim == 2:
            self.vars = [x, y, t]
            self.vars_space = [x, y]
        else:
            self.vars = [x, y, z, t]
            self.vars_space = [x, y, z]

        # Cache for symbolic expressions and lambdified functions
        self._expr_cache: dict[str, Any] = {}  # field_name -> sympy expr
        self._lambda_cache: dict[tuple[str, int], Callable] = (
            {}
        )  # (field, order) -> lambda
        self._field_types: dict[str, str] = (
            {}
        )  # field_name -> "scalar"/"vector"/"tensor"
        self._field_extra_args: dict[str, list] = (
            {}
        )  # field_name -> extra sympy symbols

    def register_field(
        self, name: str, expr_or_callable: Any, field_type: str = "vector"
    ) -> None:
        """Register a field with either a sympy expression or a callable.

        Parameters
        ----------
        name : str
            Field name (e.g., "u", "p").
        expr_or_callable : sympy expression, list of expressions, or Callable
            If sympy expression(s), will be cached and lambdified on access.
            If Callable, forwards to parent register_field.
        field_type : str, optional
            "scalar", "vector", or "tensor", by default "vector"
        """
        # If it's already a callable, forward to parent
        if callable(expr_or_callable):
            return super().register_field(name, expr_or_callable)

        # Otherwise, cache the sympy expression
        self._expr_cache[name] = expr_or_callable
        self._field_types[name] = field_type

        # Track extra (non space/time) symbols to bind via kwargs
        base_syms = set(self.vars)
        extra_syms = set()
        for expr in (
            expr_or_callable
            if isinstance(expr_or_callable, list)
            else [expr_or_callable]
        ):
            extra_syms.update(expr.free_symbols)
        extra_syms.difference_update(base_syms)
        self._field_extra_args[name] = sorted(extra_syms, key=lambda s: s.name)

        # Register a getter that will lambdify on first access
        def field_getter(time_derivative: int = 0, **kwargs):
            base = self._get_field_function(name, time_derivative)

            # Bind any provided kwargs (e.g., t or custom symbols) so callers can fix parameters
            if kwargs:

                def bound_field(x):
                    return base(x, **kwargs)

                return bound_field

            return base

        return super().register_field(name, field_getter)

    def get_field(self, field: str, time_derivative: int = 0, **kwargs) -> Callable:
        """Access field with optional time derivative.

        Parameters
        ----------
        field : str
            Field name as registered.
        time_derivative : int, optional
            Time derivative order (0 = base field), by default 0.
        **kwargs
            Forwarded to parent get_field method.

        Returns
        -------
        Callable
            Field callable with time_derivative applied.
        """
        return super().get_field(field, time_derivative=time_derivative, **kwargs)

    def _get_field_function(self, field: str, order: int) -> Callable:
        """Lazily lambdify field expression and its derivatives.

        Parameters
        ----------
        field : str
            Field name.
        order : int
            Time derivative order.

        Returns
        -------
        Callable
            Lambdified function f(x, t=0).
        """
        # Check cache first
        cache_key = (field, order)
        if cache_key in self._lambda_cache:
            lambda_func = self._lambda_cache[cache_key]

            # Return wrapped function
            def solution(x, t=0, **kwargs):
                # Collect arguments in lambdify order: spatial coords, time, then extras
                args = [*(x[i, :] for i in range(self.dim)), t]
                extras = self._field_extra_args.get(field, [])
                for sym in extras:
                    if sym.name not in kwargs:
                        raise KeyError(
                            f"Missing value for symbol '{sym.name}' in field '{field}'"
                        )
                    args.append(kwargs[sym.name])

                if self._field_types.get(field) == "scalar":
                    return lambda_func(*args)
                else:  # vector or tensor
                    results = [lf(*args) for lf in lambda_func]
                    return _stack(tuple(results))

            return solution

        # Get or compute the derivative expression
        expr = self._get_derivative_expr(field, order)

        # Lambdify it
        field_type = self._field_types.get(field, "vector")
        args = self.vars + self._field_extra_args.get(field, [])
        if field_type == "scalar":
            self._lambda_cache[cache_key] = lambdify(args, expr, "numpy")
        else:  # vector or tensor
            self._lambda_cache[cache_key] = [lambdify(args, e, "numpy") for e in expr]

        # Recursively call to get wrapped function
        return self._get_field_function(field, order)

    def _get_derivative_expr(self, field: str, order: int) -> Any:
        """Get symbolic derivative expression.

        Parameters
        ----------
        field : str
            Field name.
        order : int
            Derivative order (0 = base, 1 = d/dt, 2 = d²/dt², ...).

        Returns
        -------
        Any
            Sympy expression or list of expressions.
        """
        if field not in self._expr_cache:
            raise ValueError(f"Field '{field}' has no cached expression")

        base_expr = self._expr_cache[field]

        if order == 0:
            return base_expr
        else:
            # Differentiate base expression
            if isinstance(base_expr, list):
                return [diff(e, t, order) for e in base_expr]
            else:
                return diff(base_expr, t, order)


class AnalyticalIncompressibleNavierStokesSolution(AnalyticalSolution):
    """
    Base class for analytical incompressible Navier-Stokes solutions.
    Extends AnalyticalSolution with automatic computation of stress tensors
    and standardized field names for NS equations.

    Design Philosophy
    -----------------
    This class enforces a standard interface for incompressible NS solutions:

    1. **Standard Fields** - Automatically registers: "u" (velocity), "p" (pressure),
       "sigma" (stress tensor), "p_mean" (mean pressure)

    2. **Stress Tensor Computation** - Automatically computes the Cauchy stress
       tensor from velocity and pressure: σ = 2μS - pI, where S is the strain rate

    3. **Verification Support** - Provides confirm_incompressible_NS_solution()
       to symbolically verify the solution satisfies the NS equations

    Inheritance Guide
    -----------------
    Child classes must:

    1. **Call super().__init__(dim)** with dimension (2 or 3)

    2. **Define material properties**:
       - `self.rho` - Density (kg/m³)
       - `self.nu` - Kinematic viscosity (m²/s)
       - `self.mu` - Dynamic viscosity (Pa·s)

    3. **Define velocity and pressure** as SymPy expressions:
       - `self.u` - List of velocity components [ux, uy] or [ux, uy, uz]
       - `self.p` - Pressure expression
       - `self.p_mean` (optional) - Mean pressure (defaults to 0)

    4. **Call self._register_NS_fields()** after setting the above:
       - This triggers field registration and lambdification
       - Automatically creates "u", "p", "sigma", "p_mean" fields

    5. **Optionally verify** the solution:
       - Call `self.confirm_incompressible_NS_solution()` to check if the
         symbolic expressions satisfy continuity and momentum equations

    Registered Fields
    -----------------
    After calling _register_NS_fields(), the following fields are available:

    - **"u"** (vector): Velocity field, shape (dim, n_points)
    - **"p"** (scalar): Pressure field, shape (n_points,)
    - **"sigma"** (tensor): Cauchy stress tensor, flattened to (dim², n_points)
    - **"p_mean"** (scalar): Spatially-averaged pressure (function of time only)

    All fields support time derivatives via the time_derivative parameter.

    Stress Tensor Details
    ---------------------
    The stress tensor is computed as:

    σ_ij = 2μS_ij - pδ_ij

    where S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2 is the strain rate tensor.
    The result is flattened row-wise: [σ_xx, σ_xy, σ_yx, σ_yy] for 2D.

    Example
    -------
    >>> class TaylorGreen2D(AnalyticalIncompressibleNavierStokesSolution):
    ...     def __init__(self, L=2*np.pi, nu=1.0):
    ...         super().__init__(dim=2)
    ...         self.L = L
    ...         self.nu = self.mu = nu
    ...         self.rho = 1.0
    ...
    ...         # Define velocity and pressure symbolically
    ...         k = 2*sp.pi/L
    ...         self.u = [
    ...             sp.sin(k*x) * sp.cos(k*y) * sp.exp(-2*k**2*nu*t),
    ...             -sp.cos(k*x) * sp.sin(k*y) * sp.exp(-2*k**2*nu*t)
    ...         ]
    ...         self.p = -0.25*self.rho*(sp.cos(2*k*x) + sp.cos(2*k*y)) * sp.exp(-4*k**2*nu*t)
    ...         self.p_mean = 0
    ...
    ...         # Register all NS fields
    ...         self._register_NS_fields()
    ...
    ...         # Optionally verify solution
    ...         assert self.confirm_incompressible_NS_solution()
    >>>
    >>> tg = TaylorGreen2D()
    >>> print(tg.available_fields)  # ['u', 'p', 'sigma', 'p_mean']
    >>> coords = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> u = tg.get_field("u")(coords, t=0.5)
    >>> p = tg.get_field("p")(coords, t=0.5)
    >>> sigma = tg.get_field("sigma")(coords, t=0.5)  # Flattened 2x2 tensor
    >>> dudt = tg.get_field("u", time_derivative=1)(coords, t=0.5)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        # These will be set by child classes
        self.u: list | None = None
        self.p: Any | None = None
        self.rho: float | None = None
        self.nu: float | None = None
        self.mu: float | None = None
        self.p_mean: Any | None = None

    def _register_NS_fields(self) -> None:
        """Register u, p, sigma, p_mean fields."""
        if self.u is None or self.p is None:
            raise ValueError(
                "Child class must set self.u and self.p before calling lambdify()"
            )
        if self.rho is None or self.nu is None or self.mu is None:
            raise ValueError(
                "Child class must set self.rho, self.nu, and self.mu before calling lambdify()"
            )

        # Register velocity (vector field)
        self.register_field("u", self.u, field_type="vector")

        # Register pressure (scalar field)
        self.register_field("p", self.p, field_type="scalar")

        # Register stress tensor (computed field)
        sigma_expr = self._compute_sigma_expr()
        self.register_field("sigma", sigma_expr, field_type="tensor")

        # Register mean pressure (scalar constant or expression)
        if self.p_mean is None:
            self.p_mean = 0

        # Special handling for p_mean - it's a function of time only, not space
        p_mean_lambda = lambdify("t", self.p_mean, "numpy")
        self.register_field(
            "p_mean", lambda **kwargs: p_mean_lambda(kwargs.get("t", 0))
        )

    def _compute_sigma_expr(self) -> list:
        """Compute stress tensor from u and p.

        sigma = 2*mu*S - p*I, where S = (grad(u) + grad(u)^T)/2
        """
        # Compute velocity gradient
        dudx = [
            [diff(u_comp, x_comp) for u_comp in self.u] for x_comp in self.vars_space
        ]

        # Compute flattened stress tensor
        sigma_flat = [
            (
                self.mu * (dudx[i][j] + dudx[j][i]) - self.p
                if i == j
                else self.mu * (dudx[i][j] + dudx[j][i])
            )
            for i, j in itertools.product(range(self.dim), range(self.dim))
        ]
        return sigma_flat

    def confirm_incompressible_NS_solution(self) -> bool:
        """
        Checks whether the analytical solution satisfies the
        incompressible NS equations using on-demand derivative computation.

        Returns
        -------
        bool
            Whether incompressible NS is satisfied by this instantiation.
        """
        residual_cont = 0
        dudt = self._get_derivative_expr("u", 1)  # Get 1st time deriv of u

        for i in range(self.dim):
            xi = self.vars_space[i]
            ui = self.u[i]
            residual_mom = self.rho * dudt[i] + diff(self.p, xi)
            for j in range(self.dim):
                xj = self.vars_space[j]
                uj = self.u[j]
                residual_mom += self.rho * uj * diff(ui, xj)
                residual_mom -= self.nu * diff(diff(ui, xj), xj)
            assert simplify(residual_mom) == 0, (
                f"Residual component {i}:",
                simplify(residual_mom),
            )
            residual_cont += diff(ui, xi)
        assert simplify(residual_cont) == 0, residual_cont
        return True


def _stack(input: tuple) -> NDArray | list[NDArray]:
    """Basic numpy stack, except if one of the entries is scalar"""
    try:
        return np.stack(input)
    except (ValueError, TypeError):
        output = []
        veclen = 0
        for val in input:
            try:
                veclen = len(val)
            except TypeError:
                pass
            else:
                break
        if veclen == 0:
            raise NotImplementedError
        for val in input:
            try:
                len(val)  # raises except if scalar
                output.append(val)
            except TypeError:
                if abs(val) < 1e-15:
                    output.append(np.zeros(veclen))
                else:
                    output.append(val * np.ones(veclen))
        return output
