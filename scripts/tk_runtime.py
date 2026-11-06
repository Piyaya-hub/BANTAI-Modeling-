"""Resolve a Windows Python installation's bundled Tcl/Tk scripts safely.

Virtual environments share the base interpreter DLLs but can lose the Tcl/Tk
library environment that the standard Windows installer normally supplies.
Only absent or stale environment paths are repaired; valid caller overrides
are preserved and no installation files are changed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


def configure_tcl_tk_environment(*, base_prefix=None, environ=None):
    env = os.environ if environ is None else environ
    base = Path(base_prefix or sys.base_prefix)
    result = {}
    for variable, relative, required in (
        ("TCL_LIBRARY", Path("tcl") / "tcl8.6", "init.tcl"),
        ("TK_LIBRARY", Path("tcl") / "tk8.6", "tk.tcl"),
    ):
        current = env.get(variable)
        if current and (Path(current) / required).is_file():
            ordinary_base_path = base / relative
            try:
                is_base_library = Path(current).resolve() == ordinary_base_path.resolve()
            except OSError:
                is_base_library = False
            # Python can stat the normal path while Tcl cannot cross its
            # AppData junction. Upgrade this specific base-install path only;
            # preserve valid caller-supplied libraries elsewhere.
            if not (os.name == "nt" and is_base_library and not current.startswith("\\\\?\\")):
                result[variable] = str(Path(current))
                continue
        candidate = base / relative
        if not (candidate / required).is_file():
            raise RuntimeError(f"Cannot resolve {variable}: missing {candidate / required}")
        # Tcl's Windows filesystem bridge in this runtime cannot traverse the
        # AppData junction in the long path. The extended-length form bypasses
        # that junction while still referring to the same installed files.
        value = "\\\\?\\" + str(candidate) if os.name == "nt" else str(candidate)
        env[variable] = value
        result[variable] = value
    return result
