"""Compatibility wrapper for protection schemas."""

from services.api.schemas.protection import *  # type: ignore  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("__")]
