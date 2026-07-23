"""Python 3.12+ / NumPy 2.x compatibility layer.

Monkey-patches deprecated or removed APIs before they are imported by
chumpy, smplx, or other legacy dependencies. Must be imported BEFORE
any code that:
- loads SMPL-X model files (which may contain chumpy-backed arrays)
- imports chumpy (directly or via pickle deserialization)

Usage:
    import common.compat  # do this FIRST, before any other imports
"""

from collections import namedtuple
import inspect
import warnings
import numpy as np

# --- inspect.getargspec (removed in Python 3.11) ---
# Restore as a true compatibility wrapper for chumpy.
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(
            spec.args,
            spec.varargs,
            spec.varkw,
            spec.defaults,
        )

    inspect.getargspec = getargspec

# --- NumPy 2.x deprecated aliases ---
# NumPy 2.0 removed the Python-builtin aliases: bool, int, float, complex,
# object, str, unicode. Some packages (chumpy, etc.) still import them
# via "from numpy import bool, int, ...".
# Use try/except to avoid FutureWarning from hasattr in NumPy >= 2.1.
_ALIASES = [
    ('bool', bool),
    ('int', int),
    ('float', float),
    ('complex', complex),
    ('object', object),
    ('str', str),
    ('unicode', str),
]

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    for _name, _replacement in _ALIASES:
        if not hasattr(np, _name):
            setattr(np, _name, _replacement)

# --- chumpy <-> numpy bridge ---
# Ensure chumpy arrays can convert through np.array() without errors.
if not hasattr(np, 'bool_'):
    np.bool_ = bool
