from .reference_solution import ReferenceSolution
import dolfinx.io.gmsh as dgmsh
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import gmsh as pygmsh
from pathlib import Path

from numpy.typing import NDArray
from typing import Callable

__all__ = ["DFG2D_1", "DFG2D_2", "DFG2D_3"]


class DFG2D(ReferenceSolution):
    def __init__(self) -> None:
        super().__init__()
        self.L = 2.2
        self.H = 0.41
        self.c_x = self.c_y = 0.2  # Center of cylinder
        self.r = 0.05
        self.nu = self.mu = 0.001  # Dynamic viscosity
        self.rho = 1

        self._facets_getters = {
            "inlet": self._get_inlet_facets,
            "outlet": self._get_outlet_facets,
            "wall": self._get_wall_facets,
            "cylinder": self._get_cylinder_facets,
        }
        self._elements_getters = {}
        self._dofs_getters = {}

    def create_mesh(self, refinement_level: int = 1, order: int = 2) -> mesh.Mesh:
        """
        Creates the mesh for this particular reference problem, and stores
        it in object for interfacing with facet getters, etc.

        Parameters
        ----------
        refinement_level : int, optional
            Dofs: ... , by default 1
        order : int, optional
            Polynomial order of the isoparametric elements, by default 2

        Returns
        -------
        mesh.Mesh
            The stored mesh.
        """
        # From: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
        dim = 2

        pygmsh.initialize(interruptible=False)
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = pygmsh.model.occ.addRectangle(0, 0, 0, self.L, self.H, tag=1)
            cylinder = pygmsh.model.occ.addDisk(self.c_x, self.c_y, 0, self.r, self.r)

        if mesh_comm.rank == model_rank:
            fluid = pygmsh.model.occ.cut([(dim, rectangle)], [(dim, cylinder)])
            pygmsh.model.occ.synchronize()

        fluid_marker = 1
        if mesh_comm.rank == model_rank:
            volumes = pygmsh.model.getEntities(dim=dim)
            assert len(volumes) == 1
            pygmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
            pygmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        inlet_tag, outlet_tag, wall_tag, cylinder_tag = 2, 3, 4, 5
        self.boundary_tags = {
            "inlet": inlet_tag,
            "outlet": outlet_tag,
            "wall": wall_tag,
            "cylinder": cylinder_tag,
        }
        inflow, outflow, walls, cylinder = [], [], [], []
        if mesh_comm.rank == model_rank:
            boundaries = pygmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                center_of_mass = pygmsh.model.occ.getCenterOfMass(
                    boundary[0], boundary[1]
                )
                if np.allclose(center_of_mass, [0, self.H / 2, 0]):
                    inflow.append(boundary[1])
                elif np.allclose(center_of_mass, [self.L, self.H / 2, 0]):
                    outflow.append(boundary[1])
                elif np.allclose(
                    center_of_mass, [self.L / 2, self.H, 0]
                ) or np.allclose(center_of_mass, [self.L / 2, 0, 0]):
                    walls.append(boundary[1])
                else:
                    cylinder.append(boundary[1])
            pygmsh.model.addPhysicalGroup(1, walls, wall_tag)
            pygmsh.model.setPhysicalName(1, wall_tag, "Walls")
            pygmsh.model.addPhysicalGroup(1, inflow, inlet_tag)
            pygmsh.model.setPhysicalName(1, inlet_tag, "Inlet")
            pygmsh.model.addPhysicalGroup(1, outflow, outlet_tag)
            pygmsh.model.setPhysicalName(1, outlet_tag, "Outlet")
            pygmsh.model.addPhysicalGroup(1, cylinder, cylinder_tag)
            pygmsh.model.setPhysicalName(1, cylinder_tag, "cylinder")

        # Create distance field from cylinder.
        # Add threshold of mesh sizes based on the distance field
        # LcMax -                  /--------
        #                      /
        # LcMin -o---------/
        #        |         |       |
        #       Point    DistMin DistMax
        res_min = self.r / (3 * refinement_level)
        if mesh_comm.rank == model_rank:
            distance_field = pygmsh.model.mesh.field.add("Distance")
            pygmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", cylinder)
            threshold_field = pygmsh.model.mesh.field.add("Threshold")
            pygmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
            pygmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
            pygmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * self.H)
            pygmsh.model.mesh.field.setNumber(threshold_field, "DistMin", self.r)
            pygmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * self.H)
            min_field = pygmsh.model.mesh.field.add("Min")
            pygmsh.model.mesh.field.setNumbers(
                min_field, "FieldsList", [threshold_field]
            )
            pygmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if mesh_comm.rank == model_rank:
            pygmsh.option.setNumber("Mesh.Algorithm", 8)
            pygmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            pygmsh.option.setNumber("Mesh.RecombineAll", 1)
            pygmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            pygmsh.model.mesh.generate(dim)
            pygmsh.model.mesh.setOrder(order)
            pygmsh.model.mesh.optimize("Netgen")

        mesh_data = dgmsh.model_to_mesh(pygmsh.model, mesh_comm, model_rank, gdim=dim)
        self.domain = mesh_data.mesh
        self.facet_markers = mesh_data.facet_tags
        pygmsh.finalize()

        self.facet_markers.name = "Facet tags"
        return self.domain

    def _get_inlet_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["inlet"])

    def _get_outlet_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["outlet"])

    def _get_wall_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["wall"])

    def _get_cylinder_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["cylinder"])


class DFG2D_1(DFG2D):
    """
    Two-dimensional steady-state flow around an offset cylinder/circle.
    Metrics are the lift and drag coefficients, the time of their maximum,
    and the pressure difference between cylinder front and back.

    https://doi.org/10.1007/978-3-322-89849-4_39

    Query `available_fields` and `available_facet_getters` properties to see
    what can be accessed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_field("u_inflow", self._u_inflow)

    def _u_inflow(self) -> Callable:
        def solution(x: NDArray[np.float64]) -> NDArray[np.float64]:
            values = np.zeros(x.shape)
            values[0] = 0.3 * x[1] * (0.41 - x[1]) / (0.205**2)
            values[1] = 0
            return values

        return solution


class DFG2D_2(DFG2D):
    """
    Two-dimensional unsteady, `Re=100`, flow around an offset cylinder/
    circle, with a constant inflow profile. Metrics are the lift and drag
    coefficients, the time of their maximum, and the pressure difference
    between cylinder front and back, and the Strouhal number.

    https://doi.org/10.1007/978-3-322-89849-4_39

    Query `available_fields` and `available_facet_getters` properties to see
    what can be accessed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_field("u_inflow", self._u_inflow)

    def _u_inflow(self) -> Callable:
        def solution(x: NDArray[np.float64]) -> NDArray[np.float64]:
            values = np.zeros(x.shape)
            values[0] = 1.5 * x[1] * (0.41 - x[1]) / (0.205**2)
            values[1] = 0
            return values

        return solution


class DFG2D_3(DFG2D):
    """
    Two-dimensional unsteady, `Re=100`, flow around an offset cylinder/
    circle, with time-varying inflow profile. Metrics are the lift and drag
    coefficients, the time of their maximum, and the pressure difference
    between cylinder front and back, and the Strouhal number.

    - https://doi.org/10.1007/978-3-322-89849-4_39
    - https://doi.org/10.1002/fld.679
    - https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    - https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html

    Query `available_fields`, `available_facet_getters`, and `available_reference_data`
    properties to see what can be accessed.
    """

    def __init__(self) -> None:
        super().__init__()
        # Register inflow fields with derivative support
        self.register_field("u_inflow", self._u_inflow)

        # doi.org/10.1002/fld.679
        # Register benchmark data
        self.register_reference_data("Cd_max", 2.950921575)
        self.register_reference_data("Cd_max_t", 3.93625)
        self.register_reference_data("Cl_max", 0.47795)
        self.register_reference_data("Cl_max_t", 5.693125)
        self.register_reference_data("Delta_p", -0.1116)

        # Register timeseries data getter with level parameter
        self.register_reference_data("timeseries", self._get_timeseries_data)

        # Initialize timeseries cache
        self._timeseries_cache: dict[int, dict[str, NDArray[np.float64]]] = {}

    def _u_inflow(self, t: float = 0, time_derivative: int = 0) -> Callable:

        def solution(x, t=t, time_derivative=time_derivative):
            values = np.zeros((self.dim, x.shape[1]))
            dtfact = (-1) ** (time_derivative // 2) * (np.pi / 8) ** time_derivative
            sin_or_cos = np.sin if time_derivative % 2 == 0 else np.cos
            # fmt off
            values[0] = (
                dtfact
                * 1.5
                * x[1]
                * (0.41 - x[1])
                / (0.205**2)
                * sin_or_cos(np.pi * t / 8)
            )
            # fmt on
            return values

        return solution

    def _get_timeseries_data(self, lvl: int = 4) -> dict[str, NDArray[np.float64]]:
        """
        Get the reference timeseries data, from tu-dortmund.de/~featflow.

        Parameters
        ----------
        lvl : int, optional
            Refinement level, 1-6, by default 4

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Dictionary with keys 't', 'Cl', 'Cd', 'P0', 'P1', 'Delta_p', 'dofs'
        """
        assert lvl in range(1, 7), "Supplied lvl not in range 1-6"
        if lvl not in self._timeseries_cache:
            self._load_reference_data(lvl)
        return self._timeseries_cache[lvl]

    def _load_reference_data(self, lvl: int) -> None:
        # From https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html, accessed September 2024
        draglift = np.loadtxt(
            Path(__file__).absolute().parent
            / f"data/DFG2D3-draglift_q2_cn_lv1-6_dt4/bdforces_lv{lvl}"
        )
        pressure = np.loadtxt(
            Path(__file__).absolute().parent
            / f"data/DFG2D3-pressure_q2_cn_lv1-6_dt4/pointvalues_lv{lvl}"
        )
        self._timeseries_cache[lvl] = {
            "t": draglift[:, 1],
            "Cd": draglift[:, 3],
            "Cl": draglift[:, 4],
            "P0": pressure[:, 6],
            "P1": pressure[:, 11],
            "Delta_p": pressure[:, 6] - pressure[:, 11],
            "dofs": [702, 2704, 10608, 42016, 167232, 667264][lvl - 1],
        }
