"""Compatibility package bridging legacy `app` imports to `services.api`."""

from importlib import import_module
import sys
import types

_services_pkg = import_module("services.api")
_self = sys.modules[__name__]


class _AppProxy(types.ModuleType):
    def __getattr__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(_services_pkg, name)

    def __setattr__(self, name: str, value):
        setattr(_services_pkg, name, value)
        return super().__setattr__(name, value)


_self.__class__ = _AppProxy

for _attr in dir(_services_pkg):
    if _attr.startswith("__"):
        continue
    setattr(_self, _attr, getattr(_services_pkg, _attr))

__all__ = [name for name in dir(_self) if not name.startswith("__")]
