import numpy as np

from fx4f.reference_solutions.DFG2D_cylinder import DFG2D_1, DFG2D_2, DFG2D_3


def _set_dummy_dim(obj, dim: int = 2) -> None:
    geom = type("_Geom", (), {"dim": dim})()
    obj.domain = type("_Domain", (), {"geometry": geom})()


def test_inflow_profiles_shape_and_zero_v_component():
    x = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]])
    for cls in (DFG2D_1, DFG2D_2):
        sol = cls()
        values = sol.get_field("u_inflow")(x)
        assert values.shape == x.shape
        assert np.allclose(values[1], 0.0)
        assert np.all(values[0] >= 0.0)


def test_dfg2d3_time_derivative_scaling():
    sol = DFG2D_3()
    _set_dummy_dim(sol)
    x = np.array([[0.1, 0.1, 0.1], [0.2, 0.25, 0.3]])
    base = sol.get_field("u_inflow", t=1.0, time_derivative=0)(x)[0]
    first = sol.get_field("u_inflow", t=1.0, time_derivative=1)(x)[0]
    ratio_expected = (np.pi / 8) * (np.cos(np.pi / 8) / np.sin(np.pi / 8))
    mask = np.abs(base) > 1e-12
    assert np.allclose(first[mask] / base[mask], ratio_expected)


def test_timeseries_delta_p_consistency_and_lengths():
    sol = DFG2D_3()
    timeseries = sol.get_reference_data("timeseries", lvl=4)
    data = timeseries["Delta_p"]
    assert len(data) > 0
    p0 = timeseries["P0"]
    p1 = timeseries["P1"]
    assert np.allclose(data, p0 - p1)


def test_boundary_tags_present():
    sol = DFG2D_2()
    sol.create_mesh(refinement_level=1, order=1)
    expected = {"inlet", "outlet", "wall", "cylinder"}
    assert expected.issubset(set(sol.boundary_tags.keys()))


def test_dfg2d_facet_getters():
    """Test all facet getter methods return valid facet indices."""
    sol = DFG2D_1()
    sol.create_mesh(refinement_level=1, order=1)

    # Test each facet getter
    inlet_facets = sol.get_facets("inlet")
    assert inlet_facets.size > 0
    assert inlet_facets.dtype == np.int32 or inlet_facets.dtype == np.int64

    outlet_facets = sol.get_facets("outlet")
    assert outlet_facets.size > 0

    wall_facets = sol.get_facets("wall")
    assert wall_facets.size > 0

    cylinder_facets = sol.get_facets("cylinder")
    assert cylinder_facets.size > 0

    # Ensure facets are unique and non-overlapping
    all_facets = np.concatenate(
        [inlet_facets, outlet_facets, wall_facets, cylinder_facets]
    )
    assert len(np.unique(all_facets)) == len(all_facets), "Facets should be unique"
