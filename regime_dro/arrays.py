# arrays.py

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


def _is_cupy_array(a):
    try:
        import cupy as _cp
        return isinstance(a, _cp.ndarray) or hasattr(a, "__cuda_array_interface__")
    except Exception:
        return hasattr(a, "__cuda_array_interface__")


def asnumpy_strict(a, dtype=None, order=None):
    """
    Return a NumPy ndarray (never CuPy) from possibly-CuPy input.
    """
    import numpy as _np
    if _is_cupy_array(a):
        import cupy as _cp
        out = _cp.asnumpy(a)
    else:
        out = _np.asarray(a)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    if order in ("C", "F"):
        # numpy >= 2: array(copy=False) raises when a copy is unavoidable;
        # asarray copies only if needed, in every numpy version
        out = _np.asarray(out, dtype=out.dtype, order=order)
    return out


def asxp(a, dtype=None):
    """
    Convert to xp array (CuPy if GPU, else NumPy). If already xp, return as-is.
    """
    if _is_cupy_array(a):
        return a.astype(dtype, copy=False) if dtype is not None else a
    return xp.asarray(a, dtype=dtype)


def _to_xp(A):
    mod = getattr(A, "__module__", "")
    if mod.startswith("numpy"):
        return xp.asarray(A)
    return A
