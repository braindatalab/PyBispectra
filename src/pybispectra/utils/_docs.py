"""Documentation-related helper functions."""

import inspect
import os
import sys

import pybispectra


def linkcode_resolve(domain: str, info: dict):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    Shamelessly stolen from MNE-Python.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    # deal with our decorators properly
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    fn = os.path.relpath(fn, start=os.path.dirname(pybispectra.__file__))
    fn = "/".join(os.path.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if "dev" in pybispectra.__version__:
        kind = "main"
    else:
        kind = ".".join(pybispectra.__version__.split("."))
    return (
        f"http://github.com/braindatalab/pybispectra/tree/{kind}/src/"
        f"pybispectra/{fn}{linespec}"
    )
