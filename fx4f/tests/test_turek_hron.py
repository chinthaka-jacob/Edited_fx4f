import numpy as np

from fx4f.reference_solutions.Turek_Hron import Turek_Hron


def _set_dummy_dim(obj, dim: int = 2) -> None:
    geom = type("_Geom", (), {"dim": dim})()
    obj.domain = type("_Domain", (), {"geometry": geom})()


def test_inflow_ramp_zero_then_positive():
    sol = Turek_Hron(case=3)
    _set_dummy_dim(sol)
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    start = sol.get_field("u_inflow", t=0.0)(x)
    steady = sol.get_field("u_inflow", t=2.0)(x)
    assert np.allclose(start, 0.0)
    assert steady.shape == x.shape
    assert np.all(steady[0] >= 0.0)
    assert np.allclose(steady[1], 0.0)


def test_timeseries_delta_p_consistency_and_lengths():
    sol = Turek_Hron(case=3)
    timeseries = sol.get_reference_data("timeseries", lvl=4)
    data = timeseries["Delta_p"]
    assert len(data) > 0
    p0 = timeseries["P0"]
    p1 = timeseries["P1"]
    assert np.allclose(data, p0 - p1)


def test_boundary_and_element_tags_present():
    sol = Turek_Hron(case=2)
    expected_boundary = {"inlet", "outlet", "walls", "cylinder"}
    assert expected_boundary.issubset(set(sol.available_facet_getters))
    expected_elements = {"fluid", "solid"}
    assert expected_elements.issubset(set(sol.available_element_getters))


def test_create_mesh_produces_tags():
    sol = Turek_Hron(case=1)
    domain = sol.create_mesh(refinement_level=1, order=1)
    assert domain.topology.dim == 2
    assert sol.facet_markers is not None
    assert sol.element_markers is not None
    # ensure each expected tag exists at least once
    for name in ["inlet", "outlet", "wall", "cylinder", "interface"]:
        assert len(sol.facet_markers.find(sol.boundary_tags[name])) > 0
    for name in ["fluid", "solid"]:
        assert len(sol.element_markers.find(sol.element_tags[name])) > 0


def test_turek_hron_facet_getters():
    """Test all facet getter methods return valid facet indices."""
    sol = Turek_Hron(case=2)
    sol.create_mesh(refinement_level=1, order=1)

    # Test each facet getter
    inlet_facets = sol.get_facets("inlet")
    assert inlet_facets.size > 0
    assert inlet_facets.dtype == np.int32 or inlet_facets.dtype == np.int64

    outlet_facets = sol.get_facets("outlet")
    assert outlet_facets.size > 0

    wall_facets = sol.get_facets("walls")
    assert wall_facets.size > 0

    cylinder_facets = sol.get_facets("cylinder")
    assert cylinder_facets.size > 0


def test_turek_hron_element_getters():
    """Test all element getter methods return valid element indices."""
    sol = Turek_Hron(case=3)
    sol.create_mesh(refinement_level=1, order=1)

    # Test fluid and solid element getters
    fluid_elements = sol.get_elements("fluid")
    assert fluid_elements.size > 0
    assert fluid_elements.dtype == np.int32 or fluid_elements.dtype == np.int64

    solid_elements = sol.get_elements("solid")
    assert solid_elements.size > 0

    # Ensure elements are unique and non-overlapping
    all_elements = np.concatenate([fluid_elements, solid_elements])
    assert len(np.unique(all_elements)) == len(
        all_elements
    ), "Elements should be unique"
