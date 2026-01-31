from .reference_solution import AnalyticalIncompressibleNavierStokesSolution
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import sympy as sp
from sympy.abc import x, y, t
from numpy.typing import NDArray

__all__ = ["TaylorGreen2D"]


class TaylorGreen2D(AnalyticalIncompressibleNavierStokesSolution):
    """
    Unsteady analytical solution to the 2D incompressible Navier–Stokes
    equations. Spatially periodic 'checkerboard' pattern of clock-wise and
    counter-clockwise rotating vortices. By Taylor and Green, 1937.

    Represents exponential decay of a vortical initial state. The solution
    is constructed such that it satisfies a vector heat equation, satisfies
    `curl( u*nabla(u) ) = 0`. The latter ensures the non-linear convective
    term can be written as the gradient of a scalar, i.e., the pressure.

    https://doi.org/10.1098/rspa.1937.0036

    Query `available_fields` and `available_facet_getters` properties to see
    what can be accessed via `get_field()` and `get_facets()`.
    """

    def __init__(self, L: float = 2 * np.pi, nu: float = 1, mode: int = 1) -> None:
        """
        Default paramters produce one periodic patch of 4x4 vortices.

        Parameters
        ----------
        L : float, optional
            Domain length, by default 2*np.pi
        nu : float, optional
            Kinematic viscosity, by default 1
        mode : int, optional
            Number of vortices in length pi, by default 1
        """
        super().__init__(dim=2)
        self.L = L
        self.nu = self.mu = nu
        self.rho = 1

        self._facets_getters = {
            "domain_boundary": self._get_domain_boundary_facets,
            "walls_vertical": self._get_walls_vertical_facets,
            "walls_horizontal": self._get_walls_horizontal_facets,
        }
        self._elements_getters = {}
        self._dofs_getters = {"origin": self._get_origin_dofs}

        # fmt: off
        self.u = [
            sp.sin(2 * np.pi * x * mode / L) * sp.cos(2 * np.pi * y * mode / L) * sp.exp(-2 * nu * (2 * np.pi * mode / L) ** 2 * t),
            -sp.cos(2 * np.pi * x * mode / L) * sp.sin(2 * np.pi * y * mode / L) * sp.exp(-2 * nu * (2 * np.pi * mode / L) ** 2 * t),
        ]
        self.p = (
            0.25 * (sp.cos(2 * np.pi * 2 * x * mode / L) + sp.cos(2 * np.pi * 2 * y * mode / L) - 2)
            * sp.exp(-4 * (2 * np.pi * mode / L) ** 2 * nu * t)
        )
        self.p_mean = sp.integrate(sp.integrate(self.p, (x, 0, L)), (y, 0, L)) / L**2
        # fmt: on

        self._register_NS_fields()

    def create_mesh(self, nx: int = 16, ny: int = 16) -> mesh.Mesh:
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0, 0), (self.L, self.L)],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )
        return self.domain

    def periodic_boundary_indicator(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.logical_or(
            np.isclose(x[0], self.L), np.isclose(x[1], self.L)
        )  # Ouflow plane and right side

    # Map right to left and top to bottom
    def periodic_relation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_out = x.copy()  # Copy all coordinates
        xL_boundary_indx = np.isclose(x[0], self.L).nonzero()
        x_out[0][xL_boundary_indx] -= self.L  # Right plane to left plane
        yL_boundary_indx = np.isclose(x[1], self.L).nonzero()
        x_out[1][yL_boundary_indx] -= self.L  # Upper plane to lower plane
        return x_out

    def _get_domain_boundary_facets(self) -> NDArray[np.int64]:
        def walls(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return (
                np.isclose(x[0], 0)
                | np.isclose(x[0], self.L)
                | np.isclose(x[1], 0)
                | np.isclose(x[1], self.L)
            )

        return mesh.locate_entities_boundary(self.domain, self.dim - 1, walls)

    def _get_walls_vertical_facets(self) -> NDArray[np.int64]:
        def walls_vertical(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.isclose(x[0], 0) | np.isclose(x[0], self.L)

        return mesh.locate_entities_boundary(self.domain, self.dim - 1, walls_vertical)

    def _get_walls_horizontal_facets(self) -> NDArray[np.int64]:
        def walls_horizontal(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.isclose(x[1], 0) | np.isclose(x[1], self.L)

        return mesh.locate_entities_boundary(
            self.domain, self.dim - 1, walls_horizontal
        )

    def _get_origin_dofs(
        self,
        function_space: fem.FunctionSpace | list[fem.FunctionSpace, fem.FunctionSpace],
    ) -> NDArray[np.int64]:
        def node_origin(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

        return fem.locate_dofs_geometrical(function_space, node_origin)
