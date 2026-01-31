from .reference_solution import AnalyticalIncompressibleNavierStokesSolution
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import sympy as sp
from sympy.abc import x, y, z, t
from numpy.typing import NDArray

__all__ = ["EthierSteinman"]


class EthierSteinman(AnalyticalIncompressibleNavierStokesSolution):
    """
    Unsteady analytical solution to the incompressible Navier–Stokes
    equations that is fully three-dimensional, involving all three
    Cartesian velocity components, each of which depends non-trivially on
    all three co-ordinate directions. By Ethier and Steinman, 1994.

    Exponential decay of an initial state. Just like a Taylor-Green vortex,
    the solution is constructed such that it satisfies a vector heat
    equation, satisfies `curl( u*nabla(u) ) = 0`. The latter ensures the
    non-linear convective term can be written as the gradient of a scalar,
    i.e., the pressure.

    https://doi.org/10.1002/fld.1650190502

    Query `available_fields`, `available_facet_getters`, and `available_dof_getters`
    properties to see what can be accessed.
    """

    def __init__(
        self,
        L: float = 2,
        nu: float = 1,
        L_sol: float = 2,
        a: float = np.pi / 4,
        d: float = np.pi / 2,
    ) -> None:
        """
        Parameters
        ----------
        L : float, optional
            Cube domain (centered around origin) side size, by default 2
        nu : float, optional
            Kinematic viscosity, by default 1
        L_sol : float, optional
            Length-scale of solution, by default 2 (original paper)
        a : float, optional
            _description_, by default np.pi/4
        d : float, optional
            _description_, by default np.pi/2
        """
        super().__init__(dim=3)
        self.L = L
        self.rho = 1
        self.nu = self.mu = nu  # rho-normalized stresses
        self.a = a
        self.d = d

        self._facets_getters = {"domain_boundary": self._get_domain_boundary_facets}
        self._elements_getters = {}
        self._dofs_getters = {"origin": self._get_origin_dofs}

        # fmt: off
        l = L_sol/2 # Omega = [-l,l]^3
        self.u = [ (sp.exp(a * x / l) * sp.sin(a * y / l + d * z / l) + sp.exp(a * z / l) * sp.cos(a * x / l + d * y / l)) * sp.exp(-nu * d**2 / l**2 * t) ,\
                   (sp.exp(a * y / l) * sp.sin(a * z / l + d * x / l) + sp.exp(a * x / l) * sp.cos(a * y / l + d * z / l)) * sp.exp(-nu * d**2 / l**2 * t) ,\
                   (sp.exp(a * z / l) * sp.sin(a * x / l + d * y / l) + sp.exp(a * y / l) * sp.cos(a * z / l + d * x / l)) * sp.exp(-nu * d**2 / l**2 * t) ]
        self.p = -1/2 * (
            sp.exp(2 * a * x / l) + sp.exp(2 * a * y / l) + sp.exp(2 * a * z / l) - 3
            + 2 * sp.sin(a * x / l + d * y / l) * sp.cos(a * z / l + d * x / l) * sp.exp(a * (y + z) / l)
            + 2 * sp.sin(a * y / l + d * z / l) * sp.cos(a * x / l + d * y / l) * sp.exp(a * (z + x) / l)
            + 2 * sp.sin(a * z / l + d * x / l) * sp.cos(a * y / l + d * z / l) * sp.exp(a * (x + y) / l)
            ) * sp.exp(-2 * nu * d**2 / l**2 * t)
        if abs(a-d) < 1E-10:
            self.p_mean = -1/2 * (
                3 * L**2 * l * sp.sinh(L * a / l) / a - 3 * L**3
                + 3 * l**3 * (
                    -sp.exp(2 * L * d / l) * sp.cos(L * d / l)
                    + 2 * sp.exp(L * d / l) - sp.cos(L * d / l)
                    ) * sp.exp(-L * d / l) * sp.sin(L * d / l) / (2 * d**3)
                ) * sp.exp(-2 * nu * d**2 / l**2 * t) / L**3
        else:
            self.p_mean = -1/2 * (
                3 * L**2 * l * sp.sinh(L * a / l) / a - 3 * L**3  \
                + 3 * (
                    l**3 * (-sp.sqrt(2) * a**2 * sp.exp(2 * L * a / l) * sp.sin(L * d / l) * sp.cos(L * a / l + sp.pi / 4) 
                    + 2 * a**2 * sp.exp(L * a / l) * sp.sin(L * d / l) 
                    - sp.sqrt(2) * a**2 * sp.sin(L * d / l) * sp.sin(L * a / l + sp.pi / 4) 
                    + 2 * a * d * sp.exp(2 * L * a / l) * sp.cos(L * a / l) * sp.cos(L * d / l) 
                    - 2 * a * d * sp.exp(2 * L * a / l) 
                    - 2 * a * d * sp.cos(L * a / l) * sp.cos(L * d / l) 
                    + 2 * a * d + sp.sqrt(2) * d**2 * sp.exp(2 * L * a / l) * sp.sin(L * d / l) * sp.sin(L * a / l + sp.pi / 4) 
                    - 2 * d**2 * sp.exp(L * a / l) * sp.sin(L * d / l) 
                    + sp.sqrt(2) * d**2 * sp.sin(L * d / l) * sp.cos(L * a / l + sp.pi / 4)) * sp.exp(-L * a / l) / (a * (a**4 - d**4))
                    )
                ) * sp.exp(-2 * nu * d**2 / l**2 * t) / L**3
        # fmt: on

        self._register_NS_fields()

    def create_mesh(self, nx: int = 4, ny: int = 4, nz: int = 4) -> mesh.Mesh:
        self.domain = mesh.create_box(
            MPI.COMM_WORLD,
            [
                (-self.L / 2, -self.L / 2, -self.L / 2),
                (self.L / 2, self.L / 2, self.L / 2),
            ],
            [nx, ny, nz],
            mesh.CellType.hexahedron,
        )
        return self.domain

    def _get_domain_boundary_facets(self) -> NDArray[np.int64]:
        def walls(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return (
                np.isclose(x[0], -self.L / 2)
                | np.isclose(x[0], self.L / 2)
                | np.isclose(x[1], -self.L / 2)
                | np.isclose(x[1], self.L / 2)
                | np.isclose(x[2], -self.L / 2)
                | np.isclose(x[2], self.L / 2)
            )

        return mesh.locate_entities_boundary(self.domain, self.dim - 1, walls)

    def _get_origin_dofs(
        self,
        function_space: fem.FunctionSpace | list[fem.FunctionSpace, fem.FunctionSpace],
    ) -> NDArray[np.int64]:
        def node_origin(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.logical_and(
                np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0)),
                np.isclose(x[2], 0),
            )

        return fem.locate_dofs_geometrical(function_space, node_origin)
