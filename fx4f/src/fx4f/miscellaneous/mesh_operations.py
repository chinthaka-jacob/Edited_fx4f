import numpy as np
from functools import lru_cache
from typing import Iterable, Tuple

from mpi4py import MPI
from numpy.typing import NDArray

import dolfinx
from dolfinx import fem, default_real_type
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
from dolfinx.mesh import refine
from basix.ufl import element

import ufl

__all__ = [
    "get_element_sizes",
    "gather_coefficients",
    "get_point_values",
    "get_domain_volume",
    "sample_on_refined_linears",
]


@lru_cache(maxsize=128)
def get_domain_volume(domain: dolfinx.mesh.Mesh) -> float:
    """
    Cache the domain volume for a given mesh.

    Parameters
    ----------
    mesh : fem.FunctionSpace
        Mesh object for which to compute the volume.

    Returns
    -------
    float
        Domain volume computed via MPI allreduce.
    """
    vol = domain.comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(domain, default_real_type(1.0)) * ufl.dx)
        ),
        op=MPI.SUM,
    )
    return vol


def get_point_values(
    func: fem.Function, points: Iterable[tuple[float, float, float]]
) -> list[NDArray]:
    """
    Evaluate field at points.

    Parameters
    ----------
    func : fem.Function
        Field to be evaluated
    points : list[tuple[float,float,float]]
        list of 3d points

    Returns
    -------
    list[float]
        Point values
    """
    # Based on https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    # Also implemented in scifem, but without caching? https://github.com/scientificcomputing/scifem/blob/main/src/scifem/eval.py
    domain = func.function_space.mesh
    points = tuple(tuple(point) for point in points)  # Ensure hashable type
    point_values = []
    for i, cell_ids in enumerate(_get_point_evaluation_cell_ids(domain, points)):
        value_local = func.eval(points[i], cell_ids[:1]) if len(cell_ids) > 0 else None
        values = domain.comm.allgather(value_local)
        point_values.append(next(value for value in values if value is not None))
    return point_values


@lru_cache
def _get_point_evaluation_cell_ids(
    domain: dolfinx.mesh.Mesh, points: tuple[tuple[float, float, float]]
) -> list[NDArray]:
    """Cached cell ids for point evaluation."""
    tree = bb_tree(domain, domain.geometry.dim)
    points = np.array(points)
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
    return [colliding_cells.links(i) for i in range(len(points))]


def get_element_sizes(domain: dolfinx.mesh.Mesh) -> fem.Function:
    """
    Element sizes computed as Ve^(1/d).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh

    Returns
    -------
    fem.Function
        DG0 function with element sizes
    """
    DG0 = fem.functionspace(domain, ("DG", 0))
    h = fem.Function(DG0)
    v = ufl.TestFunction(DG0)
    Vol_form = fem.form(v * ufl.dx)
    Vol_vec = fem.assemble_vector(Vol_form)
    h.x.array[:] = Vol_vec.array[:] ** (1 / domain.geometry.dim)
    h.x.scatter_forward()
    return h


def sample_on_refined_linears(
    fields: fem.Function | list[fem.Function],
    refinement_level: int = 0,
    domain: dolfinx.mesh.Mesh | None = None,
) -> list[fem.Function]:
    """
    Samples fields on refined P1 grid. Useful for exporting higher-order
    functions in vtk format.

    Parameters
    ----------
    fields : fem.Function | list[fem.Function]
        Fields to be interpolated
    refinement_level : int, optional
        Number of element refinements, by default 0
    domain : dolfinx.mesh.Mesh | None, optional
        Mesh on which to sample, if unspecified this will be the mesh of
        the (first) field, by default None

    Returns
    -------
    list[fem.Function]
        Interpolated functions.
    """
    #
    if type(fields) is not list and type(fields) is not tuple:
        fields = [fields]
    if domain is None:
        domain = fields[0].function_space.mesh
    for r in range(refinement_level):
        raise NotImplementedError
        domain = refine(domain)
    fields_interpolated = []
    for uh in fields:
        shape = uh.ufl_shape
        refinement_level = 0  # Perform refinement only once
        We = element("Lagrange", domain.basix_cell(), 1, shape=shape)
        W = fem.functionspace(domain, We)
        u_P1 = fem.Function(W)
        u_P1.interpolate(uh)
        u_P1.name = uh.name
        fields_interpolated.append(u_P1)
    return fields_interpolated


def gather_coefficients(u: fem.Function) -> NDArray:
    """
    Obtain a single numpy array of coefficients of the given function as if
    it were on a single core.

    TODO: improve implementation, probably doesn't work for Hdiv/Hcurl
    spaces.

    Parameters
    ----------
    u : fem.Function
        FEM function to transfer to a single core.

    Returns
    -------
    np.typing.NDArray
        Numpy array of coefficients.
    """
    # Collapse to 1 core
    coeffs = u.x.array[:]
    coeffs_gathered = MPI.COMM_WORLD.gather(coeffs, root=0)
    _, non_duplicates = _gather_non_duplicate_coords(u)
    if MPI.COMM_WORLD.rank == 0:
        coeffs = np.concatenate(coeffs_gathered)
        return coeffs[non_duplicates]
    coeffs_dim = u.ufl_shape[0] if len(u.ufl_shape) else 1
    return np.zeros(coeffs_dim)


@lru_cache()
def _gather_non_duplicate_coords(
    u: fem.Function, atol: float = 10e-8
) -> Tuple[NDArray, NDArray]:
    """
    Obtain the array with coerdinates associated to the dofs in u, on a
    single cpu core. 'non-duplicate' refers to duplication in degrees of
    freedom in parallel execution.

    Parameters
    ----------
    u : fem.Function
        Function of which to obtain determine the non-duplicate
        coordinates.
    atol : float, optional
        Tolerance for making sure coordinates equal, by default 10e-8.

    Returns
    -------
    tuple[np.typing.NDArray, np.typing.NDArray]
        Numpy array of coordinates, and numpy array of indices of unique
        dofs when the dof array of different cores are concatenated.
    """
    # Collapse to 1 core
    coeffs_dim = u.ufl_shape[0] if len(u.ufl_shape) else 1
    coords = u.function_space.tabulate_dof_coordinates()
    coords_gathered = MPI.COMM_WORLD.gather(coords, root=0)
    if MPI.COMM_WORLD.rank == 0:
        coords = np.concatenate(coords_gathered)
        coords_floored = np.round(coords / atol).astype(int)
        _, non_duplicates = np.unique(coords_floored, return_index=True, axis=0)
        non_duplicates_coeffs = []
        for d in range(coeffs_dim):
            non_duplicates_coeffs.append(non_duplicates * coeffs_dim + d)
        non_duplicates_coeffs = np.ravel(non_duplicates_coeffs, order="F")
        return coords[non_duplicates], non_duplicates_coeffs
    return np.zeros((1, coords.shape[1])), np.zeros(coeffs_dim)
