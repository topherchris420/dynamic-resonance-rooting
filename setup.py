"""Compatibility shim for legacy tooling.

Project metadata lives in ``pyproject.toml``. Keeping this file minimal avoids
two competing package definitions while still supporting older setuptools flows.
"""

from setuptools import setup


setup()
