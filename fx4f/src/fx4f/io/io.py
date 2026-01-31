from pathlib import Path
from typing import Iterable, Sequence

from mpi4py import MPI
import adios4dolfinx
import numpy as np
import dolfinx.fem as fem
import dolfinx

__all__ = [
    "write_checkpoint",
    "read_mesh",
    "read_checkpoint",
    "read_checkpoint_metadata",
]


# Checkpointing:
def write_checkpoint(
    filename: str | Path,
    functions: fem.Function | Sequence[fem.Function] | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Write a checkpoint (mesh + functions + metadata) with overwrite=True.

    Parameters
    ----------
    filename : str | Path
        Path to checkpoint file/directory, e.g., outputdir+"/checkpoint.bp".
    functions : fem.Function | Sequence[fem.Function], optional
        Functions to be saved. Must be provided (cannot be None).
    metadata : dict, optional
        Metadata dict to store (e.g., time, step). Empty dict if None.

    Raises
    ------
    ValueError
        If functions is None.
    """
    if functions is None:
        raise ValueError("functions must be provided for checkpoint writing")

    f1 = functions if not type(functions) == list else functions[0]
    mesh = f1.function_space.mesh
    write_mesh(filename, mesh, overwrite=True)
    write_functions(filename, functions)
    adios4dolfinx.write_attributes(
        Path(filename),
        MPI.COMM_WORLD,
        "metadata",
        __treat_metadata_in(metadata or {}),
    )


def read_checkpoint(
    filename: str | Path,
    functions: fem.Function | Sequence[fem.Function] | None = None,
) -> None:
    """
    Load functions from a checkpoint file at time=0.

    Parameters
    ----------
    filename : str | Path
        Path to checkpoint file/directory, e.g., outputdir+"/checkpoint.bp".
    functions : fem.Function | Sequence[fem.Function], optional
        Function(s) into which to load data. Must match saved functions
        in type, space, and mesh. Must be provided (cannot be None).

    Raises
    ------
    ValueError
        If functions is None.
    """
    read_functions(filename, functions)


def read_checkpoint_metadata(filename: str | Path) -> dict:
    """
    Read the metadata stored with a checkpoint.

    Parameters
    ----------
    filename : str | Path
        Path to checkpoint file/directory, e.g., outputdir+"/checkpoint.bp"

    Returns
    -------
    dict
        Metadata that was stored, e.g., time.
    """
    return __treat_metadata_out(
        adios4dolfinx.read_attributes(Path(filename), MPI.COMM_WORLD, "metadata")
    )


def write_mesh(
    filename: str | Path, mesh: dolfinx.mesh.Mesh, overwrite: bool = False
) -> None:
    """
    Port to `adios4dolfinx` functionality for storing the mesh. Is the
    first step of storing a solution state.

    Parameters
    ----------
    filename : str | Path
        Path to file/directory, e.g., outputdir+"/solution.bp"
    mesh : dolfinx.mesh.Mesh
        Mesh to be saved
    overwrite : bool, optional
        Whether to clear the directory and rewrite mesh, by default False
    """
    filename = Path(filename)
    mode = (
        adios4dolfinx.adios2_helpers.adios2.Mode.Write
        if overwrite
        else adios4dolfinx.adios2_helpers.adios2.Mode.Append
    )
    adios4dolfinx.write_mesh(filename, mesh, mode=mode)


def write_functions(
    filename: str | Path,
    functions: fem.Function | Sequence[fem.Function] | None = None,
    time: float = 0,
) -> None:
    """
    Append function(s) to an ADIOS2 checkpoint at specified time.

    Parameters
    ----------
    filename : str | Path
        Path to file/directory, e.g., outputdir+"/solution.bp".
    functions : fem.Function | Sequence[fem.Function], optional
        Function(s) to append. Named cp0, cp1, ... internally.
        Must be provided (cannot be None).
    time : float, optional
        Time index for this state, by default 0.

    Raises
    ------
    ValueError
        If functions is None.
    """
    if functions is None:
        raise ValueError("functions must be provided for writing")

    filename = Path(filename)
    if not isinstance(functions, Iterable) or isinstance(functions, fem.Function):
        functions = [functions]  # type: ignore[list-item]

    for i, function in enumerate(functions):
        adios4dolfinx.write_function(filename, function, time=time, name=f"cp{i}")


def read_mesh(filename: str | Path) -> dolfinx.mesh.Mesh:
    """
    Port to `adios4dolfinx` functionality for loading the mesh.

    Parameters
    ----------
    filename : str | Path
        Path to file/directory, e.g., outputdir+"/solution.bp"

    Returns
    -------
    dolfinx.mesh.Mesh
        The stored mesh
    """
    filename = Path(filename)
    return adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)


def read_functions(
    filename: str | Path,
    functions: fem.Function | Sequence[fem.Function] | None = None,
    time: float = 0,
) -> None:
    """
    Load function(s) from an ADIOS2 checkpoint at specified time.

    Parameters
    ----------
    filename : str | Path
        Path to file/directory, e.g., outputdir+"/solution.bp".
    functions : fem.Function | Sequence[fem.Function], optional
        Function(s) into which to load data (cp0, cp1, ...).
        Must be provided (cannot be None).
    time : float, optional
        Time index to load, by default 0.

    Raises
    ------
    ValueError
        If functions is None.
    """
    if functions is None:
        raise ValueError("functions must be provided for reading")

    filename = Path(filename)
    if not isinstance(functions, Iterable) or isinstance(functions, fem.Function):
        functions = [functions]  # type: ignore[list-item]

    for i, function in enumerate(functions):
        adios4dolfinx.read_function(filename, function, time=time, name=f"cp{i}")


def __treat_metadata_in(metadata: dict) -> dict:
    """Helper for formatting metadata, to ensure all keys are `np.ndarrays`."""
    metadata_ = {}
    for key, val in metadata.items():
        if type(val) is not np.ndarray:
            val = np.array([val])
        metadata_[key] = val
    return metadata_


def __treat_metadata_out(metadata: dict) -> dict:
    """Helper for formatting metadata, to unpack size 1 `np.ndarrays` to
    their value."""
    metadata_ = {}
    for key, val in metadata.items():
        if val.size == 1:
            val = val[0]
        metadata_[key] = val
    return metadata_
