"""Postventa bot package with shared utilities."""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

__all__ = ["APP_DIR", "ROOT_DIR"]
