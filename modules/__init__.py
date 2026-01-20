# modules/__init__.py

# make modules a package,
# expose the 'solver' sub-package
from . import solver

__all__ = ['solver']