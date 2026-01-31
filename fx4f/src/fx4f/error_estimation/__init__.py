"""Error estimation and convergence analysis."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static imports for type checkers / language servers (Pylance)
    from .error_norms import *  # noqa: F401,F403

import pkgutil
import importlib

# Dynamically discover and import all modules at runtime
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
