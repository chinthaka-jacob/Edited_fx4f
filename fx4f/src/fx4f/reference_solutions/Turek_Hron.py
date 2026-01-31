from .reference_solution import ReferenceSolution
from dolfinx.io import gmsh as gmshio
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import gmsh
from pathlib import Path

from numpy.typing import NDArray
from typing import Callable

__all__ = ["Turek_Hron"]


class Turek_Hron(ReferenceSolution):
    """
    Fluid-structure interaction benchmark: 2D channel with elastic beam attached
    to circular cylinder. Three test cases (FSI-1, FSI-2, FSI-3) with different
    Reynolds numbers and structural properties.

    Reference: Turek & Hron, 2006
    https://doi.org/10.1007/3-540-34596-5_15

    Query `available_fields`, `available_facet_getters`, `available_element_getters`,
    and `available_reference_data` properties to see what can be accessed.
    """

    def __init__(self, case=3) -> None:
        """Parameters
        ----------
        case : int, optional
        """
        super().__init__()

        # Geometry
        self.L = 2.5
        self.H = 0.41
        self.c_x = self.c_y = 0.2  # Center of cylinder
        self.r = 0.05
        self.l = 0.35  # Elastic structure length
        self.h = 0.02  # Elastic structure thickness

        # Case specific
        self.U = 0.2 if case == 1 else 1 if case == 2 else 2
        self.rho_s = 1.0e3 if case == 1 else 10.0e3 if case == 2 else 1.0e3
        self.nu_s = 0.4  # Poissons ratio
        self.mu_s = (
            0.5e6 if case == 1 else 0.5e6 if case == 2 else 2.0e6
        )  # Shear modulus
        self.lambda_s = self.mu_s * (2 * self.nu_s / (1 - 2 * self.nu_s))
        self.rho_f = 1e3
        self.nu_f = 1e-3
        self.mu_f = self.nu_f * self.rho_f

        self.register_field("u_inflow", self._u_inflow)

        self._facets_getters = {
            "inlet": self._get_inlet_facets,
            "outlet": self._get_outlet_facets,
            "walls": self._get_wall_facets,
            "cylinder": self._get_cylinder_facets,
        }

        self._elements_getters = {
            "fluid": self._get_fluid_elements,
            "solid": self._get_solid_elements,
        }

        self._dofs_getters = {}

        # Register benchmark data
        self.register_reference_data("pointA", (0.6, 0.2))
        self.register_reference_data("pointb", (0.2, 0.2))

        # Register timeseries data getter with level parameter
        self.register_reference_data("timeseries", self._get_timeseries_data)

        # Initialize timeseries cache
        self._timeseries_cache: dict[int, dict[str, NDArray[np.float64]]] = {}

    def _u_inflow(self, t: float = 0) -> Callable:

        def solution(x, t=t):
            values = np.zeros((self.dim, x.shape[1]))
            t_ramp = (1 - np.cos(np.pi / 2 * t)) / 2 if t < 2 else 1
            values[0] = t_ramp * 1.5 * self.U * x[1] * (0.41 - x[1]) / (0.205**2)
            return values

        return solution

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
        # Based on: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
        dim = 2

        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        fluid_tag, solid_tag = 1, 2
        self.element_tags = {
            "fluid": fluid_tag,
            "solid": solid_tag,
        }
        inlet_tag, outlet_tag, wall_tag, cylinder_tag, interface_tag = 1, 2, 3, 4, 5
        self.boundary_tags = {
            "inlet": inlet_tag,
            "outlet": outlet_tag,
            "wall": wall_tag,
            "cylinder": cylinder_tag,
            "interface": interface_tag,
        }

        gmsh.initialize(interruptible=False)
        if mesh_comm.rank == model_rank:
            gmsh.model.occ.addRectangle(0, 0, 0, self.L, self.H, tag=10)
            gmsh.model.occ.addRectangle(
                self.c_x,
                self.c_y - self.h / 2,
                0,
                self.r + self.l,
                self.h,
                tag=20,
            )
            gmsh.model.occ.addDisk(self.c_x, self.c_y, 0, self.r, self.r, tag=99)
            gmsh.model.occ.cut([(dim, 10), (dim, 20)], [(dim, 99)])
            gmsh.model.occ.synchronize()

            volumes = gmsh.model.getEntities(dim=dim)
            print(volumes)
            assert (
                len(volumes) == 2
            )  # Rectangle minus cylinder and solid minux cylinder
            gmsh.model.addPhysicalGroup(dim, [volumes[1][1]], tag=fluid_tag)
            gmsh.model.addPhysicalGroup(dim, [volumes[0][1]], tag=solid_tag)

            inflow, outflow, walls, cylinder, interface = [], [], [], [], []
            if mesh_comm.rank == model_rank:
                for boundary in gmsh.model.getBoundary((volumes[1],), oriented=False):
                    center_of_mass = gmsh.model.occ.getCenterOfMass(
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
                    elif (center_of_mass[0] - self.c_x) ** 2 + (
                        center_of_mass[1] - self.c_y
                    ) ** 2 < self.r**2 + 1e-7:
                        cylinder.append(boundary[1])
                for boundary in gmsh.model.getBoundary((volumes[0],), oriented=False):
                    center_of_mass = gmsh.model.occ.getCenterOfMass(
                        boundary[0], boundary[1]
                    )
                    if (center_of_mass[0] - self.c_x) ** 2 + (
                        center_of_mass[1] - self.c_y
                    ) ** 2 > self.r**2 + 1e-7:
                        interface.append(boundary[1])
            gmsh.model.addPhysicalGroup(1, walls, tag=wall_tag)
            gmsh.model.addPhysicalGroup(1, inflow, tag=inlet_tag)
            gmsh.model.addPhysicalGroup(1, outflow, tag=outlet_tag)
            gmsh.model.addPhysicalGroup(1, cylinder, tag=cylinder_tag)
            gmsh.model.addPhysicalGroup(1, interface, tag=interface_tag)

            # Create distance field from cylinder.
            # Add threshold of mesh sizes based on the distance field
            # LcMax -                  /--------
            #                      /
            # LcMin -o---------/
            #        |         |       |
            #       Point    DistMin DistMax
            res_min = self.r / refinement_level
            res_max = self.H / refinement_level

            field_distance_cylinder_tag = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                field_distance_cylinder_tag, "EdgesList", cylinder
            )
            field_threshold_cylinder_tag = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(
                field_threshold_cylinder_tag, "IField", field_distance_cylinder_tag
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_cylinder_tag, "LcMin", res_min
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_cylinder_tag, "LcMax", res_max
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_cylinder_tag, "DistMin", self.r
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_cylinder_tag, "DistMax", 2 * self.H
            )

            field_distance_interface_tag = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                field_distance_interface_tag, "EdgesList", interface
            )
            field_threshold_interface_tag = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(
                field_threshold_interface_tag, "IField", field_distance_interface_tag
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_interface_tag, "LcMin", res_min
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_interface_tag, "LcMax", res_max
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_interface_tag, "DistMin", self.r
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold_interface_tag, "DistMax", 2 * self.H
            )

            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(
                min_field,
                "FieldsList",
                [field_threshold_cylinder_tag, field_threshold_interface_tag],
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.generate(dim)
            gmsh.model.mesh.setOrder(order)
            gmsh.model.mesh.optimize("Netgen")

        mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=dim)
        self.domain = mesh_data.mesh
        self.element_markers: mesh.MeshTags = mesh_data.cell_tags
        self.facet_markers: mesh.MeshTags = mesh_data.facet_tags
        gmsh.finalize()

        return self.domain

    def _get_fluid_elements(self) -> NDArray[np.int64]:
        return self.element_markers.find(self.element_tags["fluid"])

    def _get_solid_elements(self) -> NDArray[np.int64]:
        return self.element_markers.find(self.element_tags["solid"])

    def _get_inlet_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["inlet"])

    def _get_outlet_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["outlet"])

    def _get_wall_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["wall"])

    def _get_cylinder_facets(self) -> NDArray[np.int64]:
        return self.facet_markers.find(self.boundary_tags["cylinder"])

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
        # From https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html, accessed December 2024
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
