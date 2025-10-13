"""Compatibility wrapper bridging legacy imports to `services.api.ingesta_unificada`."""

from importlib import import_module
import sys
import types

_impl = import_module('services.api.ingesta_unificada')
_self = sys.modules[__name__]

class _Proxy(types.ModuleType):
    def __getattr__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(_impl, name)

    def __setattr__(self, name: str, value):
        setattr(_impl, name, value)
        return super().__setattr__(name, value)

_self.__class__ = _Proxy

for _name in dir(_impl):
    if _name.startswith('__'):
        continue
    setattr(_self, _name, getattr(_impl, _name))



__all__ = [name for name in dir(_self) if not name.startswith('__')]
