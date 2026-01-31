from .reference_solution import AnalyticalIncompressibleNavierStokesSolution
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import sympy as sp
from sympy.abc import x, y, t
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict

__all__ = ["John2004_1"]

class John2004_1(AnalyticalIncompressibleNavierStokesSolution):
    """
    Implementation of the manufactured solution from John (2004).
    
    This tests time-stepping schemes for the incompressible Navier-Stokes 
    equations on the unit square [0,1]×[0,1] with homogeneous Dirichlet boundary 
    conditions.

    """

    def __init__(self, L: float = 1.0, nu: float = 1.0, mode: int = 1) -> None:
        """
        Initialize the John2004 solution.

        Parameters
        ----------
        L : float, optional
            Domain length (unit square), default is 1.0
        nu : float, optional
            Kinematic viscosity (nu = 1/Re), default is 1.0
        mode : int, optional
            Reserved for future use (different solution modes), default is 1
        """
        super().__init__(dim=2)
        self.L = L
        self.nu = self.mu = nu
        self.rho = 1

        # Domain boundary getter
        self._facets_getters = {
            "domain_boundary": self._get_domain_boundary_facets,
        }
        self._elements_getters = {}
        self._dofs_getters = {"origin": self._get_origin_dofs}

        # --- Analytical solution from the paper ---
        # u1 = sin(t) * sin(pi*x) * sin(pi*y)
        # u2 = sin(t) * cos(pi*x) * cos(pi*y)
        # p  = sin(t) * ( sin(pi*x) + cos(pi*y) - 2/pi )
        self.u = [
            sp.sin(t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            sp.sin(t) * sp.cos(sp.pi * x) * sp.cos(sp.pi * y),
        ]
        self.p = sp.sin(t) * (sp.sin(sp.pi * x) + sp.cos(sp.pi * y) - 2 / sp.pi)

        # Mean pressure integral over domain
        self.p_mean = sp.integrate(sp.integrate(self.p, (x, 0, self.L)), (y, 0, self.L)) / self.L ** 2

        # --- Forcing term computed symbolically for the 
        # incompressible Navier-Stokes equations. 
        # f = du/dt + (u·∇)u + ∇p - (1/Re) * Δu
        u1 = self.u[0]
        u2 = self.u[1]
        # Time derivatives
        u1_t = sp.diff(u1, t)
        u2_t = sp.diff(u2, t)
        # Advective terms (u·∇)u_i
        conv1 = u1 * sp.diff(u1, x) + u2 * sp.diff(u1, y)
        conv2 = u1 * sp.diff(u2, x) + u2 * sp.diff(u2, y)
        # Pressure gradients
        px = sp.diff(self.p, x)
        py = sp.diff(self.p, y)
        # Viscous terms
        lap1 = sp.diff(u1, x, 2) + sp.diff(u1, y, 2)
        lap2 = sp.diff(u2, x, 2) + sp.diff(u2, y, 2)

        f1 = sp.simplify(u1_t + conv1 + px - self.nu * lap1)
        f2 = sp.simplify(u2_t + conv2 + py - self.nu * lap2)
        self.f = [f1, f2]

        # Register the analytical fields with the base class helper
        self._register_NS_fields()

        # Initialize reference data cache for time-stepping method comparisons
        self._figure_data_cache: Dict[int, Dict[str, NDArray[np.float64]]] = {}
        self.register_reference_data("figure_data", self._get_figure_data)

    def create_mesh(self, nx: int = 64, ny: int = 64) -> mesh.Mesh:
        """
        Create a rectangular mesh for the unit square domain.

        Defaults chosen to match the reference: nx=ny=64 (h = 1/64).
        """
        # h = L / nx for a square grid
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0, 0), (self.L, self.L)],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )
        return self.domain

    def _get_domain_boundary_facets(self) -> NDArray[np.int64]:
        def walls(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return (
                np.isclose(x[0], 0)
                | np.isclose(x[0], self.L)
                | np.isclose(x[1], 0)
                | np.isclose(x[1], self.L)
            )

        return mesh.locate_entities_boundary(self.domain, self.dim - 1, walls)


    def _get_origin_dofs(
        self,
        function_space: fem.FunctionSpace | list[fem.FunctionSpace, fem.FunctionSpace],
    ) -> NDArray[np.int64]:
        def node_origin(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

        return fem.locate_dofs_geometrical(function_space, node_origin)

    def _get_figure_data(self, fig: int = 1, method: str = "") -> Dict[str, NDArray[np.float64]]:
        """
        Return reference time-series error data from John2004 paper figures.
        
        The paper presents comparisons of various time-stepping methods including:
        - BWE: Backward Euler
        - CN: Crank-Nicolson
        - FS0, FS1: Fractional-step schemes
        - ROS3P, ROWDAIND2, ROS3Pw, ROS34PW2, ROS34PW3: Rosenbrock methods
        
        Data files should be stored in `data/John2004/` and named as 
        `fig{fig}_{method}.txt` where each file contains time-series data 
        (e.g., timestep vs error norm).
        
        Parameters
        ----------
        fig : int, optional
            Figure number (1 for Re=1, 2 for Re=1000), default is 1
        method : str, optional
            Specific time-stepping method to retrieve. If empty, returns all 
            available methods for the figure. Examples: "BWE", "CN", "FS0", etc.
        
        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary mapping method names to their data arrays. If method is 
            specified, returns single-item dict with that method's data.
            
        Example
        -------
        >>> refsol = John2004_1(nu=1.0)
        >>> # Get all data for Figure 1 (Re=1)
        >>> all_methods = refsol.get_reference_data("figure_data", fig=1)
        >>> # Get specific method data
        >>> cn_data = refsol.get_reference_data("figure_data", fig=1, method="CN")
        """
        assert fig in (1, 2), "Only figures 1 and 2 are available"
        if fig not in self._figure_data_cache:
            self._load_figure_data(fig)

        if method:
            if method not in self._figure_data_cache[fig]:
                raise KeyError(f"Method '{method}' not found for fig {fig}")
            return {method: self._figure_data_cache[fig][method]}

        return self._figure_data_cache[fig]

    def _load_figure_data(self, fig: int) -> None:
        """
        Load all time-stepping method data files matching `fig{fig}_*.txt`.
        
        Expected file format: Plain text files readable by np.loadtxt, typically
        containing columns [timestep, velocity_error, pressure_error] vs. time step.
        
        Parameters
        ----------
        fig : int
            Figure number to load data for (1 or 2)
        """
        data_dir = Path(__file__).absolute().parent / "data" / "John2004"
        cache: Dict[str, NDArray[np.float64]] = {}
        if not data_dir.exists():
            # No data directory yet; keep empty cache (user can add files)
            self._figure_data_cache[fig] = cache
            return

        for path in sorted(data_dir.glob(f"fig{fig}_*.txt")):
            method = path.stem.split("_")[1]
            try:
                arr = np.loadtxt(path)
            except Exception:
                # Skip malformed files but continue loading others
                continue
            cache[method] = arr

        self._figure_data_cache[fig] = cache