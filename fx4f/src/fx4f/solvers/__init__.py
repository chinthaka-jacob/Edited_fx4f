"""Time integration and linear/nonlinear solver configurations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static imports for language servers (Pylance)
    from .IMEX_Runge_Kutta import *  # noqa: F401,F403
    from .ksp_options import *  # noqa: F401,F403
    from .ksp_options_stokes import *  # noqa: F401,F403
    from .snes_options import *  # noqa: F401,F403

import pkgutil
import importlib

# Dynamically discover and import all solver configuration modules.
# This allows each module to self-register its solvers (e.g., via registries)
# without requiring explicit imports here.
_modules = [
    importlib.import_module(f".{modname}", package=__name__)
    for importer, modname, ispkg in pkgutil.iter_modules(__path__)
    if not modname.startswith("_")
]

# Re-export everything from submodules
__all__ = []
for module in _modules:
    if hasattr(module, "__all__"):
        for name in module.__all__:
            __all__.append(name)
            globals()[name] = getattr(module, name)
