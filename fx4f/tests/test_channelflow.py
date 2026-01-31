import numpy as np

from fx4f.reference_solutions.channelflow import Channelflow180


def test_dns_data_loaded_and_sorted():
    sol = Channelflow180()
    required_keys = [
        "dns_y+",
        "dns_U",
        "dns_u_rms",
        "dns_v_rms",
        "dns_w_rms",
        "dns_P",
        "dns_p_rms",
    ]
    for key in required_keys:
        arr = sol.get_reference_data(key)
        assert arr is not None
        assert len(arr) > 0
    assert np.all(np.diff(sol.get_reference_data("dns_y+")) > 0)


def test_u_init_shapes_and_finiteness():
    sol = Channelflow180()
    x = np.vstack(
        [
            np.linspace(0.0, sol.Lx, 4),
            np.linspace(-0.5 * sol.Ly, 0.5 * sol.Ly, 4),
            np.linspace(-0.5 * sol.Lz, 0.5 * sol.Lz, 4),
        ]
    )
    values = sol.get_field("u_init")(x)
    assert values.shape == x.shape
    assert np.all(np.isfinite(values))
    assert np.any(np.abs(values[0]) > 0)


def test_periodic_relation_maps_outflow_and_right_boundary():
    sol = Channelflow180()
    x = np.array(
        [
            [sol.Lx, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5 * sol.Lz],
        ],
        dtype=float,
    )
    mapped = sol.periodic_relation(x.copy())
    assert np.isclose(mapped[0, 0], 0.0)  # outflow wrapped to inflow
    assert np.isclose(mapped[2, 2], -0.5 * sol.Lz)  # right to left
    assert np.allclose(mapped[:, 1], x[:, 1])  # interior unchanged


def test_mesh_creation_stretching_and_walls_facets():
    sol = Channelflow180()
    domain = sol.create_mesh(nx=2, ny=2, nz=2, mesh_stretching=1.5)
    assert domain.topology.dim == 3
    facets = sol.get_facets("walls")
    assert facets.size > 0
    # Ensure stretching moved some y-coordinates off the linear grid
    y_coords = np.unique(np.round(domain.geometry.x[:, 1], decimals=6))
    assert len(y_coords) > 2


def test_periodic_indicator_and_corner_node():
    sol = Channelflow180()
    x = np.array(
        [
            [sol.Lx, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5 * sol.Lz],
        ]
    )
    mask = sol.periodic_boundary_indicator(x)
    assert mask[0] and mask[2] and not mask[1]


def test_channelflow_facet_getters():
    """Test facet getter methods return valid facet indices."""
    from dolfinx import fem

    sol = Channelflow180()
    sol.create_mesh(nx=2, ny=2, nz=2)

    # Test walls facet getter
    wall_facets = sol.get_facets("walls")
    assert wall_facets.size > 0
    assert wall_facets.dtype == np.int32 or wall_facets.dtype == np.int64


def test_channelflow_dof_getters():
    """Test DOF getter methods return valid DOF indices."""
    from dolfinx import fem

    sol = Channelflow180()
    sol.create_mesh(nx=2, ny=2, nz=2)

    # Create a function space to test DOF getter
    V = fem.functionspace(sol.domain, ("Lagrange", 1))

    # Test corner DOF getter (call directly with function_space parameter)
    corner_dofs = sol._get_corner_dofs(V)
    assert corner_dofs.size > 0
    assert corner_dofs.dtype == np.int32 or corner_dofs.dtype == np.int64
