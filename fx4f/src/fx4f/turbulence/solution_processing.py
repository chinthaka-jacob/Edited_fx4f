import numpy as np
from functools import lru_cache
from dolfinx import fem
from fx4f.miscellaneous.mesh_operations import (
    gather_coefficients,
    _gather_non_duplicate_coords,
)

__all__ = [
    "planar_average",
    "indices_repeating_coordinates",
]


def planar_average(
    u: fem.Function,
) -> tuple[np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Assuming a box domain, take the average of `u` in the plane
    perpendicular to the free remaining axis. Used for collecting
    turbulence statistics.

    TODO: Add axis argument

    Parameters
    ----------
    u : fem.Function
        Function of which to take the average.

    Returns
    -------
    tuple[np.typing.NDArray, list[np.typing.NDArray]]
    Array with distinct, ordered, remaining free parameter, and array with
    planar averaged values.
    """
    coeffs_dim = u.ufl_shape[0] if len(u.ufl_shape) else 1
    coeffs = gather_coefficients(u)
    y_coords, indice_maps, repeats = indices_repeating_coordinates(u, axis=1)
    all_avg_vals = []
    for d in range(coeffs_dim):
        avg_vals = np.zeros(y_coords.shape)
        for y_index in range(len(y_coords)):
            where = np.where(indice_maps == y_index)[0]
            avg_vals[y_index] = (
                np.sum(coeffs[where * coeffs_dim + d]) / repeats[y_index]
            )
        all_avg_vals.append(avg_vals[y_coords.argsort()])
    y_sort = np.sort(y_coords)
    return y_sort, all_avg_vals


@lru_cache()
def indices_repeating_coordinates(
    u: fem.Function, axis: int = 0, atol: float = 10e-8
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """
    Returns an array of non-repeating coordinates along a given axis, as
    well as an array of the size of the mesh-nodes of `u` to which index
    the first array's coordinate belongs, and for each unique coordinate
    how many points in the mesh correspond to that

    Parameters
    ----------
    u : fem.Function
        Function of which to obtain determine the non-duplicate
        coordinates.
    axis : int, optional
        (0:x, 1:y or 2:z)
    atol : float, optional
        Tolerance for making sure coordinates equal, by default 10e-8.

    Returns
    -------
    tuple[np.typing.NDArray, np.typing.NDArray]
    """
    coords, _ = _gather_non_duplicate_coords(u, atol)
    coords_floored = np.round(coords[:, axis] / atol).astype(int)
    _, indc, indc_inv, reps = np.unique(
        coords_floored, return_index=True, return_inverse=True, return_counts=True
    )
    return coords[indc, axis], indc_inv, reps
