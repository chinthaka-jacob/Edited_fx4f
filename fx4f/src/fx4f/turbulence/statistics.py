from dolfinx import fem
from basix.ufl import element
import numpy as np

__all__ = ["TurbulenceStatistics"]


class TurbulenceStatistics:
    """
    Systematism for computing turbulence statistics based on sequentially
    supplied solution states. This is implemented in such a way that not
    all solution states are actually stored.

    Supported statistics:

    - *mean*: mean of the solution fields (i.e., `<u>`)
    - *rms*: root mean square of the solution fields (i.e.,
      `sqrt(< u' u' >)` with `u' = u-<u>`)
    - *rms2*: mean square of the solution fields (e.g., `< u' u' >`)

    To keep track of statistics of different fields (velocity, pressure),
    one should instantiate different objects.

    Usage:

    >>> stats = TurbulenceStatistics('mean','rms')
    >>> stats.add_frame(u1)
    >>> stats.add_frame(u2)
    >>> stats.add_frame(u3)
    >>> u_mean = stats.mean
    >>> u_rms2 = stats.rms2

    NOTE: Implementation of statistics is performed with the variable name
    `u`, but applies to arbitrary fields.

    Attributes
    ----------
    mean : fem.Function
        Mean of all provided solution states.
    rms2 : fem.Function
        Square of the root mean square of all provided solution states,
        exactly interpolated on a space with double the polynomial degree
        as the provided solutions, as rms involves taking squares.
    rms : fem.Function
        Root mean square of all provided solution states, computed on
        a space with double the polynomial degree as the provided
        solutions, as rms involves taking squares. Note that this is only
        exact on the nodes, as the square root operation does not map to
        a space of polynomials.

    Methods
    -------
    add_frame(solution: fem.Function)
        Add a solution state.
    """

    def __init__(self, *statistics: list[str]) -> None:
        """
        Parameters
        ----------
        *statistics : list[str]
            Currently implemented: 'mean', 'rms'
        """
        self.statistics = statistics
        self.__update_map = {
            "mean": self.__update_mean,
            "rms": self.__update_rms,
        }

    @property
    def mean(self) -> fem.Function:
        self.U.x.array[:] = self.u_sum / self.u_sum_count
        return self.U

    @property
    def rms(self) -> fem.Function:
        u_rms2 = self.u_rms2
        self.u_rms.x.array[:] = np.sqrt(u_rms2.x.array[:])
        return self.u_rms

    @property
    def rms2(self) -> fem.Function:
        self.U_2p.interpolate(self.mean)
        self.u_rms2.x.array[:] = (
            self.u_squared_sum / self.u_squared_sum_count - self.U_2p.x.array[:] ** 2
        )
        return self.u_rms2

    def add_frame(self, solution: fem.Function) -> None:
        """
        Add a solution state, from which to infer statistics upon later
        query.

        Parameters
        ----------
        solution : fem.Function
            FEM function from which to infer statistics.
        """
        for stat in self.statistics:
            if stat in self.__update_map.keys():
                self.__update_map[stat](solution)

    def __update_mean(self, solution: fem.Function) -> None:
        """
        Called after every `add_frame` if activated through supplying the
        `mean` string during initialization.

        Parameters
        ----------
        solution : fem.Function
            FEM function from which to infer statistics.
        """
        if not hasattr(self, "U"):
            # First call
            self.U = _create_function(solution)
            self.u_sum = solution.x.array.copy()
            self.u_sum_count = 1
            return
        self.u_sum += solution.x.array
        self.u_sum_count += 1

    def __update_rms(self, solution: fem.Function) -> None:
        """
        Called after every `add_frame` if activated through supplying the
        `rms` string during initialization.

        Parameters
        ----------
        solution : fem.Function
            FEM function from which to infer statistics.
        """
        if not hasattr(self, "u_rms"):
            # First call
            self.u_rms = _create_function(solution, degree_raise_fact=2)
            self.u_rms2 = _create_function(solution, degree_raise_fact=2)
            self.U_2p = _create_function(
                solution, degree_raise_fact=2
            )  # Mean on 2*poldeg space
            self.u2 = _create_function(
                solution, degree_raise_fact=2
            )  # Square solution fields on 2*poldeg space
            self.u2.interpolate(solution)
            self.u_squared_sum = self.u2.x.array**2
            self.u_squared_sum_count = 1
            return
        self.u2.interpolate(solution)
        self.u_squared_sum += self.u2.x.array**2
        self.u_squared_sum_count += 1


def _create_function(uh: fem.Function, degree_raise_fact: int = 0) -> fem.Function:
    """
    Create an 'empty' function of the same type as `uh`, potentially with a
    raised polynomial degree.

    Parameters
    ----------
    uh : fem.Function
        Reference function
    degree_raise_fact : int, optional
        Raise of polynomial order, by default 0

    Returns
    -------
    fem.Function
        Function in raised polynomial space.
    """
    if degree_raise_fact == 0:
        W = uh.function_space
    else:
        degree = uh.function_space.ufl_element().degree
        family = uh.function_space.ufl_element().family_name
        shape = uh.ufl_shape
        domain = uh.function_space.mesh
        We = element(
            family, domain.basix_cell(), degree * degree_raise_fact, shape=shape
        )
        W = fem.functionspace(domain, We)
    return fem.Function(W)
