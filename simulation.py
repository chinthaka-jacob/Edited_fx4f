import numpy as np
import pickle

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fx4f", "src"))
from fx4f.reference_solutions.John2004 import John2004_1
from fx4f.fluid_mechanics import kinetic_energy
from fx4f.error_estimation import error_norm
from fx4f.solvers import KSPSetterRegistry, SNESSetterRegistry
from fx4f.io import io as fx4fio

from treelog4dolfinx import (
    treelog4dolfinx,
    log, log_info, 
    log_figure_pyvista, log_figure_mpl, 
    LogTime, LogCounter)
import pyvista4dolfinx as p4d

from utils.plotting import plot_energy_evolution

from dolfinx import fem, io
from basix.ufl import element, mixed_element

import ufl
from ufl import div, grad, nabla_grad, dot, inner, outer, sym

from mpi4py import MPI
from petsc4py import PETSc

@treelog4dolfinx # Wrap in logging environment and handle argument parsing
def main(outputdir:str = "", # Directory for storing output
         Re:float = 1, # Reynolds number (as in kinematic viscosity)
         T:float = 1, # Final time
         dt: float = 0.05, # Time step
         nx:int = 16, # Number of elements in width direction
         ny:int = 16, # Number of elements in height direction
         Pu:int = 2, # Polynomial order velocity
         Pp:int = 2, # Polynomial order pressure
         Ct:float = 4.0, # Stabilization coefficient, time
         Cd:float = 36.0, # Stabilization coefficient, diffusion
         Ca:float = 1., # Stabilization coefficient, advection
         q_degree:int = 4, # Quadrature degree
         showplots:bool = False, # Whether to show plots on screen or to just export to file
    ):
    """
    Two-dimensional equal-order stabilized unsteady Navier-Stokes simulation 
    of the John2004 test case with Dirichlet boundary conditions.
    
    Expected output for run with default parameter values and Re=1, T=1:
    Reference solution from John et al., 2004.
    """

    # Reference solution
    nu = 1/Re
    RefSol = John2004_1(L=1.0, nu=nu)

    # Create square quad mesh
    with LogTime("Mesh generation"):
        domain = RefSol.create_mesh(nx, ny)

    # Element sizes
    hx  = RefSol.L/nx
    hy  = RefSol.L/ny

    # Zero bodyforce (John2004 includes forcing terms in the solution)
    force = fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0)))
    
    # Create function spaces
    with LogTime("Function spaces") as LT:
        Ve = element("Lagrange", domain.basix_cell(), Pu, shape=(domain.geometry.dim,))
        Qe = element("Lagrange", domain.basix_cell(), Pp)
        W_el = mixed_element([Ve, Qe])
        W = fem.functionspace(domain, W_el)
        V, WV_map = W.sub(0).collapse(); LT.log_time("Collapse V") # DOF map from W to V
        Q, WQ_map = W.sub(1).collapse(); LT.log_time("Collapse Q") # DOF map from W to Q
    
    # Pressure at corner point condition (zero mean pressure constraint)
    dofs_corner = RefSol.get_dofs("origin", (W.sub(1), Q))
    bc_zero_p = fem.dirichletbc(fem.Function(Q), dofs_corner, W.sub(1))

    # Dirichlet boundary conditions on domain walls (non-periodic)
    # Get boundary facets and apply zero velocity
    boundary_facets = RefSol.get_facets("domain_boundary")
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_u_wall = fem.locate_dofs_topological((W.sub(0), V), domain.geometry.dim - 1, boundary_facets)
    bc_u_wall = fem.dirichletbc(u_zero, dofs_u_wall, W.sub(0))

    # Collect all Dirichlet boundary conditions
    bcs = [bc_zero_p, bc_u_wall]

    # Define trial and test functions
    w_trial = ufl.TrialFunction(W)
    v, q = ufl.TestFunctions(W)

    # Create functions for solution and intermediate storage
    dudt_p_new = fem.Function(W)
    dudt_new, p_new = ufl.split(dudt_p_new)
    dudt_prev, u_new, u_prev, p_new_flat = fem.Function(V), fem.Function(V), fem.Function(V), fem.Function(Q)

    # Stabilization parameters
    G  = ufl.as_matrix( [[4.0/hx**2, 0], [0, 4.0/hy**2]] )
    GG = 16.0/hx**4+16.0/hy**4
    gg = 4.0/hx**2+4.0/hy**2
    tausupg = ( Ct/dt**2 + Ca*dot(u_prev,dot(G, u_prev)) + Cd*nu**2*GG )**(-0.5)
    taupspg = tausupg
    taulsic = 1/(gg*tausupg)
    
    # Basic generalized alpha time stepping parameters
    rho_inf = 0.5
    alpha_m	= 0.5*(3.0-rho_inf)/(1.0+rho_inf)
    alpha_f	= 1.0/(1.0+rho_inf)
    gamma	= 0.5 + alpha_m - alpha_f

    # Dependency intermediate state on previous and new solution
    dudt = dudt_prev + alpha_m * ( dudt_new-dudt_prev )
    dudt_res = dudt_prev + alpha_f * ( dudt_new-dudt_prev )
    u = u_prev + alpha_f * dt * ( dudt_prev+gamma*(dudt_new-dudt_prev) )
    p = p_new

    # Residuals: momentum, continuity and boundary
    resm  = dudt_res + dot(u,nabla_grad(u)) - nu*div(nabla_grad(u)) + grad(p) - force
    resc  = div(u)

    # Specify quadrature degree for subsquent integrations
    dx = ufl.Measure("dx", domain, metadata={'quadrature_degree': q_degree})

    # Strain operator
    def epsilon(u):
        return sym(nabla_grad(u))
    
    # Navier-Stokes weak form
    B =  dot( dudt		     , v             )*dx \
     - inner( outer(u,u)       , nabla_grad(v) )*dx \
       + dot( q              , div(u)        )*dx \
       - dot( div(v)         , p             )*dx \
     + inner( 2*nu*epsilon(u), epsilon(v)    )*dx \
       - dot( force		     , v             )*dx

    # Stabilization
    B_SUPG = dot( dot(u,nabla_grad(v)), tausupg*resm )*dx \
         + dot( dot(u,grad(v))        , tausupg*resm )*dx \
       - inner( nabla_grad(v)	      , outer(tausupg*resm,tausupg*resm) )*dx
    B_PSPG = dot( nabla_grad(q)       , taupspg*resm )*dx 
    B_LSIC = inner( div(v)            , taulsic*resc )*dx
    B_stab = B_SUPG+B_PSPG+B_LSIC

    # Combined form and its jacobian
    F = B + B_stab
    J = ufl.derivative(F, dudt_p_new, w_trial)

    # Set up the nonlinear Newton solver (without MPC for non-periodic)
    problem = fem.petsc.NonlinearProblem(F, dudt_p_new, bcs=bcs, J=J, petsc_options_prefix="nlvs_")
    SNESSetterRegistry.get("default")(problem.solver, log_iterations=True) # Nonlinear solver settings
    KSPSetterRegistry.get("direct")(problem.solver, log_iterations=True) # Linear solver settings

    # Output VTK
    u_new.name, p_new_flat.name = "u", "p"
    u_file = io.VTXWriter(domain.comm, f"{outputdir}/u.bp", [u_new])
    p_file = io.VTXWriter(domain.comm, f"{outputdir}/p.bp", [p_new_flat])

    # Initial condition
    t = 0.0
    log_counter_timestep = LogCounter("Timestep")
    log_info(f'Time: {t}')
    RefSol.interpolate_field(u_new, 'u', t=t)
    RefSol.interpolate_field(dudt_prev, 'u', t=t, time_derivative=1)
    RefSol.interpolate_field(p_new_flat, 'p', t=t)
    dudt_p_new.x.array[WV_map] = dudt_prev.x.array[:]
    log("Interpolated initial conditions.")

    # Plotting IC
    plotter = p4d.Plotter(off_screen=not showplots)
    p4d.plot(u_new, plotter=plotter)
    p4d.plot(p_new_flat, plotter=plotter)
    log_figure_pyvista("u_p.png", plotter)

    # Store energy evolution for postprocessing:
    u_ref = fem.Function(V)
    RefSol.interpolate_field(u_ref, 'u', t=t)
    energydata = {'E': [kinetic_energy(u_ref)], 'Eh': [kinetic_energy(u_new)]}
    timesteps = [t]
    
    # Time-stepping
    while t < T - 0.5*dt:
        timestep = log_counter_timestep.next()
        t += dt
        log_info('Time: {:.{decimals}f}'.format(t, decimals=len(str(dt).split(".")[1]))) # Log the time with number of decimals equal to that of dt
 
        # Move last solutions to the `previous' variables
        dudt_prev.x.array[:] = dudt_p_new.x.array[WV_map]
        u_prev.x.array[:] = u_new.x.array[:]

        # Nonlinear iterations to determine new dudt and p
        with LogTime("Solving", title_upon_exit=True):
            problem.solve()
            
        # Compute new velocity from previous u and new and previous dudt
        u_new.x.array[:] += dudt_prev.x.array[:]*(dt*(1-gamma)) + dudt_p_new.x.array[WV_map]*(dt*gamma)
        p_new_flat.x.array[:] = dudt_p_new.x.array[WQ_map]

        # Post process
        LT = LogTime() 
        plotter = p4d.Plotter(off_screen=not showplots)
        p4d.plot(u_new, plotter=plotter)
        p4d.plot(p_new_flat, plotter=plotter)
        log_figure_pyvista("u_p.png", plotter); LT.log_time()
        log(f"L2 error u: {( u_L2 := error_norm(u_new, RefSol.get_field('u', t=t)) )}"); LT.log_time()
        log(f"H10 error u: {( u_H10 := error_norm(u_new, RefSol.get_field('u', t=t), norm='H10') )}"); LT.log_time()
        log(f"L2 error p: {( p_L2 := error_norm(p_new_flat, RefSol.get_field('p', t=t)) )}"); LT.log_time()
        log(f"H10 error p: {( p_H10 := error_norm(p_new_flat, RefSol.get_field('p', t=t), norm='H10') )}"); LT.log_time()

        # Obtain reference solutions for analysis
        RefSol.interpolate_field(u_ref, 'u', t=t)
        energydata['E'].append(kinetic_energy(u_ref))
        energydata['Eh'].append(kinetic_energy(u_new))
        timesteps.append(t)
        log_figure_mpl("Energy.png", plot_energy_evolution(timesteps, **energydata)); LT.log_time()
        
        # Save to file
        u_file.write(t)
        p_file.write(t)
        LT.log_time("Exported solutions")

        # Checkpoint solution
        fx4fio.write_checkpoint(outputdir+"/checkpoint.bp", functions=[dudt_p_new, u_new], metadata={'t': t})
        LT.log_time("Saved checkpoint")

    log_counter_timestep.exit()

    # Store data for postprocessing
    postprocessing_data = {"Re": Re, "dt": dt, "u_L2": u_L2, "u_H10": u_H10, "p_L2": p_L2, "p_H10": p_H10}
    with open(outputdir+'/postprocessdata.pickle', 'wb') as handle:
        pickle.dump(postprocessing_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()