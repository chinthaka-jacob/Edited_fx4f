from .reference_solution import ReferenceSolution
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
from pathlib import Path

from numpy.typing import NDArray
from typing import Callable

__all__ = ["Channelflow", "Channelflow180"]


class Channelflow(ReferenceSolution):
    """
    Turbulent channel-flow reference cases. Conventions from Vreman & Kuerten.

    https://doi.org/10.1063/1.4861064

    Some useful relations:

    Friction velocity: u_tau = sqrt( nu dU/dy )
    Viscous lengthscale: delta_nu = nu/u_tau
    Wall unit": y+ = y/delta_nu = u_tau*y/nu
    Friction Reynolds number: Re_tau = (Ly/2)*u_tau/nu

    From the balance of momentum: f*V = nu*dU/dy*2S -> f = (2/Ly)*nu*dU/dy = (2/Ly)*u_tau**2
    So, u_tau = sqrt( Ly/2*f ) and then Re_tau = sqrt( (Ly/2)**3*f ) / nu

    Taking Ly=2 and f=1 (i.e., u_tau=1) simplifies this to:
    nu = 1/Re_tau
    y+ = Re_tau*y  ->  -Re_tau < y+ < Re_tau

    From DNS we know U+_max is approximately 20, almost irrespective of Re.
    And thus U_max is approximately 22.5*u_tau = 20

    CFL then corresponds to: CFL = Umax*dt/Lx/(nx*P)
    (cannot easily be computed in wall units as u_tau does not represent delta_nu/t+)
    """

    def __init__(self, Lx: float, Ly: float, Lz: float, Re_tau: float) -> None:

        super().__init__()
        self.Lx = Lx
        self.Ly = Ly  # Typically H = Ly/2 = 1
        self.Lz = Lz
        self.Re_tau = Re_tau
        self.nu = self.mu = (Ly / 2) ** (
            3 / 2
        ) / Re_tau  # See above, applies for unit force
        self.rho = 1
        self.force = (1.0, 0.0, 0.0)

        self._facets_getters = {"walls": self._get_walls_facets}
        self._elements_getters = {}
        self._dofs_getters = {"corner": self._get_corner_dofs}

    def create_mesh(
        self, nx: int = 16, ny: int = 16, nz: int = 16, mesh_stretching: float = 0
    ) -> mesh.Mesh:
        """
        Box mesh with potential tanh mesh stretching towards the walls.

        Parameters
        ----------
        nx : int, optional
            Elements in x, by default 16
        ny : int, optional
            Elements in x, by default 16
        nz : int, optional
            Elements in x, by default 16
        mesh_stretching : float, optional
            Typically 0 or 2.75, according to Morinishi & Vasilyec 2001, by
            default 0

        Returns
        -------
        mesh.Mesh
            Box mesh with potential stretching.
        """

        Point1 = (0, -0.5 * self.Ly, -0.5 * self.Lz)
        Point2 = (self.Lx, 0.5 * self.Ly, 0.5 * self.Lz)
        self.domain = mesh.create_box(
            MPI.COMM_WORLD, [Point1, Point2], [nx, ny, nz], mesh.CellType.hexahedron
        )
        if mesh_stretching > 0:
            self.domain.geometry.x[:, 1] = (
                self.Ly
                / 2
                * np.tanh(mesh_stretching * 2 * self.domain.geometry.x[:, 1] / self.Ly)
                / np.tanh(mesh_stretching)
            )
        return self.domain

    def periodic_boundary_indicator(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.logical_or(
            np.isclose(x[0], self.Lx), np.isclose(x[2], 0.5 * self.Lz)
        )  # Ouflow plane and right side

    # Map outflow to inflow and right to left
    def periodic_relation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_out = x.copy()  # Copy all coordinates
        outflow_boundary_indx = np.isclose(x[0], self.Lx).nonzero()
        x_out[0][outflow_boundary_indx] -= self.Lx  # Right plane to left plane
        right_boundary_indx = np.isclose(x[2], 0.5 * self.Lz).nonzero()
        x_out[2][right_boundary_indx] -= self.Lz  # Upper plane to lower plane
        return x_out

    def _get_walls_facets(self) -> NDArray[np.int64]:
        def walls(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.logical_or(
                np.isclose(x[1], -0.5 * self.Ly), np.isclose(x[1], 0.5 * self.Ly)
            )

        return mesh.locate_entities_boundary(self.domain, self.dim - 1, walls)

    def _get_corner_dofs(
        self,
        function_space: fem.FunctionSpace | list[fem.FunctionSpace, fem.FunctionSpace],
    ) -> NDArray[np.int64]:
        def corner_node(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return np.logical_and(
                np.logical_and(
                    np.isclose(x[0], 0.5 * self.Lx), np.isclose(x[1], -0.5 * self.Ly)
                ),
                np.isclose(x[2], 0.5 * self.Lz),
            )

        return fem.locate_dofs_geometrical(function_space, corner_node)


class Channelflow180(Channelflow):
    """
    Turbulent channel-flow reference cases. Conventions and data from
    Vreman & Kuerten. Same domain as Moser. Vreman starts gathering
    statistics after `10*(Ly/2)/u_tau = 10s`.

    Re_tau = 180
    Re approx 6500

    - Vreman Kuerten, 2014: https://doi.org/10.1063/1.4861064
    - Kim Moin Moser, 1987: https://doi.org/10.1017/S0022112087000892
    - Moser Kim Mansour, 1999: https://doi.org/10.1063/1.869966

    Query `available_fields`, `available_facet_getters`, and `available_reference_data`
    properties to see what can be accessed.
    """

    def __init__(self) -> None:
        # Conventions and defaults from Vreman & Kuerten (doi.org/10.1063/1.4861064).
        #
        #
        super().__init__(Lx=4 * np.pi, Ly=2, Lz=4 / 3 * np.pi, Re_tau=180)
        self.register_field("u_init", self._u_init)

        # Register benchmark data
        self.register_reference_data("U_center", 18.28)
        self.register_reference_data("u_rms_center", 0.7971)
        self.register_reference_data("v_rms_center", 0.6166)
        self.register_reference_data("w_rms_center", 0.6140)

        # Register DNS data getters
        self.register_reference_data("dns_y+", lambda: self._get_dns_field("y+"))
        self.register_reference_data("dns_U", lambda: self._get_dns_field("U"))
        self.register_reference_data("dns_u_rms", lambda: self._get_dns_field("u_rms"))
        self.register_reference_data("dns_v_rms", lambda: self._get_dns_field("v_rms"))
        self.register_reference_data("dns_w_rms", lambda: self._get_dns_field("w_rms"))
        self.register_reference_data("dns_P", lambda: self._get_dns_field("P"))
        self.register_reference_data("dns_p_rms", lambda: self._get_dns_field("p_rms"))

        # Initialize DNS data storage
        self._dns_data_cache = None

    def _get_dns_field(self, field: str) -> NDArray:
        """Get a specific field from DNS data, loading if necessary."""
        if self._dns_data_cache is None:
            self._load_dns_data()
        return self._dns_data_cache[field]

    def _load_dns_data(self) -> None:
        # http://www.vremanresearch.nl/Chan180_S2.html
        # More data exists in the files in /data/channelflow180 than is exposed here,
        # and even more data is available on the above website
        self._dns_data_cache = {}
        for field in ["u", "v", "w", "p"]:
            raw_data = np.loadtxt(
                Path(__file__).absolute().parent
                / f"data/channelflow180/Chan180_S2_basic_{field}.txt",
                comments="%",
            )
            self._dns_data_cache[f"{field.capitalize()}"] = raw_data[:, 1]
            self._dns_data_cache[f"{field}_rms"] = raw_data[:, 2]
        self._dns_data_cache["y+"] = raw_data[:, 0]

    # fmt: off
    def _u_init(self) -> Callable:
        # Per Vreman 2014, doi.org/10.1016/j.jcp.2014.01.035
        # Roughly matches mean shape of U, with div-free fluctuations and roughly match magnitude of mean(u'u')
        def solution(x:NDArray[np.float64]) -> NDArray[np.float64]:
            values = np.zeros(x.shape)
            Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
            xx, yy, zz = x[0], x[1]+Ly/2, x[2]
            a, b = 22.5, 2.25
            values[0] = (
                a * yy * ( Ly - yy )
                + b * np.cos( 2 * np.pi * xx / Lx ) * np.sin( 2 * np.pi * yy / Ly ) * np.sin( 2 * np.pi * zz / Lz )
                + b * np.cos( 4 * np.pi * xx / Lx ) * np.sin( 4 * np.pi * yy / Ly ) * np.sin( 4 * np.pi * zz / Lz )
            )
            values[1] = (
                - b * Ly/(2*Lx) * np.sin( 2 * np.pi * xx / Lx ) * ( np.cos( 2 * np.pi * yy / Ly ) - 1 ) * np.sin( 2 * np.pi * zz / Lz )
                - b * Ly/(2*Lx) * np.sin( 4 * np.pi * xx / Lx ) * ( np.cos( 4 * np.pi * yy / Ly ) - 1 ) * np.sin( 4 * np.pi * zz / Lz )
            )
            values[2] = (
                - b * Lz/(2*Lx) * np.sin( 2 * np.pi * xx / Lx ) * np.sin( 2 * np.pi * yy / Ly ) * np.cos( 2 * np.pi * zz / Lz )
                - b * Lz/(2*Lx) * np.sin( 4 * np.pi * xx / Lx ) * np.sin( 4 * np.pi * yy / Ly ) * np.cos( 4 * np.pi * zz / Lz )
            )
            return values
        return solution
    # fmt: on


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     RefSol = Channelflow180()
#     dns_data = RefSol.dns_data

#     fig, axs = plt.subplots(2, 2)
#     axs[0, 0].plot(dns_data["U"], dns_data["y+"], "-r", label="U")
#     axs[0, 0].legend()
#     axs[0, 1].plot(dns_data["u_rms"], dns_data["y+"], "-r", label="u_rms")
#     axs[0, 1].plot(dns_data["v_rms"], dns_data["y+"], "-g", label="v_rms")
#     axs[0, 1].plot(dns_data["w_rms"], dns_data["y+"], "-b", label="w_rms")
#     axs[0, 1].legend()
#     axs[1, 0].plot(dns_data["P"], dns_data["y+"], "-k", label="P")
#     axs[1, 0].legend()
#     axs[1, 1].plot(dns_data["p_rms"], dns_data["y+"], "-k", label="p_rms")
#     axs[1, 1].legend()

#     fig, ax = plt.subplots(1, 1)
#     ax.semilogx(dns_data["y+"], dns_data["U"], "-r", label="U")
#     ax.legend()

#     plt.show()
