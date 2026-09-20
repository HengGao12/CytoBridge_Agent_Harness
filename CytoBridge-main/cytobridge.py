"""Lowercase import compatibility for the canonical :mod:`CytoBridge` package.

The compatibility layer is a module instead of a second package directory so
the source tree and built wheel do not contain paths that differ only by case.
That distinction matters on the default macOS and Windows filesystems.
"""

from importlib import import_module
import sys


_CANONICAL_NAME = "CytoBridge"
_PUBLIC_SUBMODULES = ("pp", "tl", "pl", "utils")
_canonical = import_module(_CANONICAL_NAME)

# Make this module package-like for callers that inspect or extend __path__.
__path__ = _canonical.__path__
if __spec__ is not None:
    __spec__.submodule_search_locations = list(__path__)

__all__ = getattr(_canonical, "__all__", _PUBLIC_SUBMODULES)

# CytoBridge imports its public package tree eagerly. Register matching aliases
# so mixed-case imports share module objects instead of executing modules twice.
for _module_name, _module in tuple(sys.modules.items()):
    if _module_name == _CANONICAL_NAME or _module_name.startswith(
        f"{_CANONICAL_NAME}."
    ):
        _alias = f"{__name__}{_module_name[len(_CANONICAL_NAME):]}"
        sys.modules.setdefault(_alias, _module)

for _name in _PUBLIC_SUBMODULES:
    globals()[_name] = import_module(f"{_CANONICAL_NAME}.{_name}")


def __getattr__(name):
    return getattr(_canonical, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_canonical)))
