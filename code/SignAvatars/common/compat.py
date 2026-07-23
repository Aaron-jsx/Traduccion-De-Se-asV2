"""Python 3.12+ / NumPy 2.x compatibility layer.

Monkey-patches deprecated or removed APIs before they are imported by
chumpy, smplx, or other legacy dependencies. Must be imported BEFORE
any code that:
- loads SMPL-X model files (which may contain chumpy-backed arrays)
- imports chumpy (directly or via pickle deserialization)

Usage:
    import common.compat  # do this FIRST, before any other imports
"""

import inspect
import warnings
import numpy as np

# --- inspect.getargspec (removed in Python 3.11) ---
# Restore as an alias of getfullargspec for chumpy 0.70 and older.
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# --- NumPy 2.x deprecated aliases ---
# NumPy 2.0 removed: np.bool, np.int, np.float, np.complex, np.object, np.str
# Use try/except to avoid FutureWarning from hasattr in NumPy >= 2.1
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    for _name, _replacement in [
        ('bool', bool),
        ('int', int),
        ('float', float),
        ('complex', complex),
        ('object', object),
        ('str', str),
    ]:
        if not hasattr(np, _name):
            setattr(np, _name, _replacement)

# --- chumpy <-> numpy bridge ---
# Ensure chumpy arrays can convert through np.array() without errors.
if not hasattr(np, 'bool_'):
    np.bool_ = bool
