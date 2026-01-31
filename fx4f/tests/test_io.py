"""Tests for the io.checkpoint helpers (adios4dolfinx-backed IO)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem, mesh


@pytest.fixture(scope="session")
def io_mod():
    """Load io/io.py as a module without clashing with built-in io."""
    module_path = (
        Path(__file__).resolve().parent.parent / "src" / "fx4f" / "io" / "io.py"
    )
    spec = importlib.util.spec_from_file_location("checkpoint_io", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for type checkers
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unit_spaces():
    """Small 2D mesh with scalar function spaces (two copies for multi-state)."""
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [(0.0, 0.0), (1.0, 1.0)],
        [2, 2],
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    V2 = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V, V2


def test_checkpoint_round_trip_single_function_metadata(io_mod, unit_spaces, tmp_path):
    domain, V, _ = unit_spaces
    fn = fem.Function(V)
    fn.x.array[:] = 2.5

    metadata = {"time": 1.23, "step": 4}
    checkpoint_path = tmp_path / "checkpoint_single.bp"

    io_mod.write_checkpoint(checkpoint_path, fn, metadata=metadata)

    fn_loaded = fem.Function(V)
    io_mod.read_checkpoint(checkpoint_path, fn_loaded)

    assert np.allclose(fn.x.array, fn_loaded.x.array)

    meta_loaded = io_mod.read_checkpoint_metadata(checkpoint_path)
    assert meta_loaded["time"] == pytest.approx(metadata["time"])
    assert meta_loaded["step"] == metadata["step"]


def test_round_trip_multiple_functions_and_times(io_mod, unit_spaces, tmp_path):
    domain, V, V2 = unit_spaces
    checkpoint_path = tmp_path / "checkpoint_multi.bp"

    io_mod.write_mesh(checkpoint_path, domain, overwrite=True)

    f0 = fem.Function(V)
    f0.x.array[:] = 1.0
    g0 = fem.Function(V2)
    g0.x.array[:] = 2.0
    io_mod.write_functions(checkpoint_path, [f0, g0], time=0.0)

    f1 = fem.Function(V)
    f1.x.array[:] = 3.0
    g1 = fem.Function(V2)
    g1.x.array[:] = 4.0
    io_mod.write_functions(checkpoint_path, [f1, g1], time=1.0)

    f0_loaded = fem.Function(V)
    g0_loaded = fem.Function(V2)
    io_mod.read_functions(checkpoint_path, [f0_loaded, g0_loaded], time=0.0)
    assert np.allclose(f0_loaded.x.array, f0.x.array)
    assert np.allclose(g0_loaded.x.array, g0.x.array)

    f1_loaded = fem.Function(V)
    g1_loaded = fem.Function(V2)
    io_mod.read_functions(checkpoint_path, [f1_loaded, g1_loaded], time=1.0)
    assert np.allclose(f1_loaded.x.array, f1.x.array)
    assert np.allclose(g1_loaded.x.array, g1.x.array)


def test_write_checkpoint_with_empty_functions_raises(io_mod, tmp_path):
    checkpoint_path = tmp_path / "checkpoint_empty.bp"
    with pytest.raises(IndexError):
        io_mod.write_checkpoint(checkpoint_path, functions=[])


def test_read_checkpoint_into_mismatched_space_raises(io_mod, unit_spaces, tmp_path):
    domain, V, _ = unit_spaces
    checkpoint_path = tmp_path / "checkpoint_mismatch.bp"

    fn = fem.Function(V)
    fn.x.array[:] = 0.5
    io_mod.write_checkpoint(checkpoint_path, fn, metadata={"time": 0.0})

    V_high = fem.functionspace(domain, ("Lagrange", 2))
    fn_mismatched = fem.Function(V_high)

    with pytest.raises(Exception):
        io_mod.read_checkpoint(checkpoint_path, fn_mismatched)


def test_read_checkpoint_metadata_missing_file_raises(io_mod, tmp_path):
    checkpoint_path = tmp_path / "does_not_exist.bp"
    with pytest.raises(Exception):
        io_mod.read_checkpoint_metadata(checkpoint_path)


def test_metadata_array_handling(io_mod, unit_spaces, tmp_path):
    _, V, _ = unit_spaces
    fn = fem.Function(V)
    fn.x.array[:] = 1.0

    metadata = {
        "arr": np.array([1.0, 2.0]),
        "list_val": [3, 4],
        "scalar": 5,
    }

    checkpoint_path = tmp_path / "checkpoint_metadata.bp"
    io_mod.write_checkpoint(checkpoint_path, fn, metadata=metadata)

    meta_loaded = io_mod.read_checkpoint_metadata(checkpoint_path)
    assert isinstance(meta_loaded["arr"], np.ndarray)
    assert np.allclose(meta_loaded["arr"], metadata["arr"])
    assert isinstance(meta_loaded["list_val"], np.ndarray)
    assert np.allclose(meta_loaded["list_val"], np.array(metadata["list_val"]))
    assert meta_loaded["scalar"] == metadata["scalar"]
