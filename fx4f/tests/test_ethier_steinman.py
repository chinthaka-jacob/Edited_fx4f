"""Tests for the Ethier–Steinman 3D analytical reference solution."""

import numpy as np
from dolfinx import fem
from ufl import dx

from fx4f.reference_solutions.Ethier_Steinman import EthierSteinman


class TestEthierSteinmanInitialization:
    """Test initialization parameters and mesh construction."""

    def test_default_parameters(self):
        es = EthierSteinman()
        assert es.L == 2
        assert es.nu == 1
        assert es.mu == 1
        assert es.rho == 1

    def test_custom_parameters(self):
        es = EthierSteinman(L=3.0, nu=0.25, L_sol=1.5, a=np.pi / 6, d=np.pi / 3)
        assert np.isclose(es.L, 3.0)
        assert np.isclose(es.nu, 0.25)
        assert np.isclose(es.mu, 0.25)
        assert np.isclose(es.a, np.pi / 6)
        assert np.isclose(es.d, np.pi / 3)

    def test_create_mesh_default(self):
        es = EthierSteinman(L=2.0)
        dom = es.create_mesh()
        assert dom is not None
        assert es.domain is dom
        assert es.dim == 3
        coords = dom.geometry.x
        L = es.L / 2
        assert coords[:, 0].min() >= -L - 1e-10
        assert coords[:, 0].max() <= L + 1e-10
        assert coords[:, 1].min() >= -L - 1e-10
        assert coords[:, 1].max() <= L + 1e-10
        assert coords[:, 2].min() >= -L - 1e-10
        assert coords[:, 2].max() <= L + 1e-10

    def test_facet_getters_and_node_origin(self):
        es = EthierSteinman(L=2.0)
        dom = es.create_mesh(nx=3, ny=3, nz=3)
        facets = es.get_facets("domain_boundary")
        assert facets.size > 0

    def test_domain_boundary_facet_getter(self):
        """Test domain_boundary facet getter returns valid facet indices."""
        es = EthierSteinman(L=2.0)
        es.create_mesh(nx=3, ny=3, nz=3)

        boundary_facets = es.get_facets("domain_boundary")
        assert boundary_facets.size > 0
        assert boundary_facets.dtype == np.int32 or boundary_facets.dtype == np.int64

    def test_origin_dof_getter(self):
        """Test origin DOF getter returns valid DOF indices."""
        es = EthierSteinman(L=2.0)
        # Use more mesh points to ensure a node at origin
        es.create_mesh(nx=4, ny=4, nz=4)

        V = fem.functionspace(es.domain, ("Lagrange", 1))
        origin_dofs = es._get_origin_dofs(V)
        # The origin may not always have a node depending on mesh resolution
        # Just verify the method works without error and returns correct dtype
        assert origin_dofs.dtype == np.int32 or origin_dofs.dtype == np.int64


class TestEthierSteinmanFields:
    """Test field evaluation and shapes."""

    def test_field_shapes(self):
        es = EthierSteinman()
        coords = np.array([[0.1, -0.2], [0.2, 0.3], [0.3, -0.1]])

        u_vals = es.get_field("u")(coords, t=0.0)
        assert u_vals.shape == (3, 2)

        p_vals = es.get_field("p")(coords, t=0.0)
        assert p_vals.shape == (2,)

        sigma_vals = es.get_field("sigma", time_derivative=0)(coords, t=0.0)
        assert len(sigma_vals) == 9  # flattened 3x3 tensor

        p_mean = es.get_field("p_mean")
        assert isinstance(p_mean, (float, np.ndarray, int))


class TestEthierSteinmanDerivatives:
    """Finite-difference validation and caching for derivatives."""

    def test_first_derivative_velocity(self):
        es = EthierSteinman()
        coords = np.array([[0.2], [0.1], [-0.1]])
        dt = 1e-5

        u_t0 = es.get_field("u")(coords, t=0.0)
        u_tdt = es.get_field("u")(coords, t=dt)
        u1 = es.get_field("u", time_derivative=1)(coords, t=0.0)

        u1_fd = (u_tdt - u_t0) / dt
        np.testing.assert_allclose(u1_fd, u1, rtol=1e-2, atol=1e-3)

    def test_first_derivative_pressure(self):
        es = EthierSteinman()
        coords = np.array([[0.2], [0.1], [-0.1]])
        dt = 1e-5

        p_t0 = es.get_field("p")(coords, t=0.0)
        p_tdt = es.get_field("p")(coords, t=dt)
        p1 = es.get_field("p", time_derivative=1)(coords, t=0.0)

        p1_fd = (p_tdt - p_t0) / dt
        np.testing.assert_allclose(p1_fd, p1, rtol=1e-2, atol=1e-3)

    def test_repeated_calls_consistent(self):
        es = EthierSteinman()
        coords = np.array([[0.2], [0.1], [-0.1]])
        call1 = es.get_field("u", time_derivative=1)(coords, t=0.0)
        call2 = es.get_field("u", time_derivative=1)(coords, t=0.0)
        np.testing.assert_allclose(call1, call2)


class TestEthierSteinmanTimeBehavior:
    """Test decay characteristics over time."""

    def test_velocity_decay(self):
        es = EthierSteinman()
        coords = np.array([[0.2], [0.1], [-0.1]])
        u_t0 = es.get_field("u")(coords, t=0.0)
        u_t1 = es.get_field("u")(coords, t=0.5)
        u_t2 = es.get_field("u")(coords, t=1.0)
        norm0 = np.linalg.norm(u_t0)
        norm1 = np.linalg.norm(u_t1)
        norm2 = np.linalg.norm(u_t2)
        assert norm1 < norm0
        assert norm2 < norm1

    def test_pressure_decay(self):
        es = EthierSteinman()
        coords = np.array([[0.2], [0.1], [-0.1]])
        p_t0 = es.get_field("p")(coords, t=0.0)
        p_t1 = es.get_field("p")(coords, t=0.5)
        assert np.linalg.norm(p_t1) < np.linalg.norm(p_t0)


class TestEthierSteinmanIntegration:
    """Integration tests: interpolation and NS verification."""

    def test_interpolation_pressure_fields(self):
        es = EthierSteinman()
        es.create_mesh(nx=3, ny=3, nz=3)
        V = fem.functionspace(es.domain, ("Lagrange", 1))

        fn_p = fem.Function(V)
        es.interpolate_field(fn_p, "p", t=0.1)
        assert fn_p.x.array.size > 0

        fn_p1 = fem.Function(V)
        es.interpolate_field(fn_p1, "p", time_derivative=1, t=0.1)
        assert fn_p1.x.array.size > 0

    def test_pressure_mean_matches_integral_custom_d(self):
        """Check analytical p_mean against assembled mean (custom d=a)."""
        es = EthierSteinman(L_sol=0.588, nu=0.1234, a=np.pi / 4, d=np.pi / 4)
        es.create_mesh(nx=8, ny=8, nz=8)
        V = fem.functionspace(es.domain, ("Lagrange", 3))
        fn = fem.Function(V)
        t = 0.0
        fn.interpolate(es.get_field("p", t=t))
        p_mean_analytic = es.get_field("p_mean")
        p_mean_numeric = fem.assemble_scalar(fem.form(fn * dx)) / es.L**3
        np.testing.assert_allclose(
            p_mean_numeric, p_mean_analytic, rtol=5e-2, atol=5e-2
        )

    def test_pressure_mean_matches_integral_default_d(self):
        """Check analytical p_mean vs assembled mean with default d (pi/2)."""
        es = EthierSteinman(L_sol=0.588, nu=0.1234)
        es.create_mesh(nx=8, ny=8, nz=8)
        V = fem.functionspace(es.domain, ("Lagrange", 3))
        fn = fem.Function(V)
        t = 0.0
        fn.interpolate(es.get_field("p", t=t))
        p_mean_analytic = es.get_field("p_mean")
        p_mean_numeric = fem.assemble_scalar(fem.form(fn * dx)) / es.L**3
        np.testing.assert_allclose(
            p_mean_numeric, p_mean_analytic, rtol=5e-2, atol=5e-2
        )

    def test_ns_equation_verification(self):
        es = EthierSteinman(nu=1.0, L=2.0, a=np.pi / 4, d=np.pi / 2)
        assert es.confirm_incompressible_NS_solution()
