import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.tk_runtime import configure_tcl_tk_environment


class TclTkRuntimeTests(unittest.TestCase):
    def test_repairs_missing_paths_using_base_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "tcl/tcl8.6").mkdir(parents=True)
            (base / "tcl/tcl8.6/init.tcl").write_text("", encoding="utf-8")
            (base / "tcl/tk8.6").mkdir(parents=True)
            (base / "tcl/tk8.6/tk.tcl").write_text("", encoding="utf-8")
            env = {"TCL_LIBRARY": "missing", "TK_LIBRARY": ""}
            result = configure_tcl_tk_environment(base_prefix=base, environ=env)
            prefix = "\\\\?\\" if os.name == "nt" else ""
            self.assertEqual(result["TCL_LIBRARY"], prefix + str(base / "tcl/tcl8.6"))
            self.assertEqual(result["TK_LIBRARY"], prefix + str(base / "tcl/tk8.6"))
            self.assertEqual(env["TCL_LIBRARY"], result["TCL_LIBRARY"])

    def test_preserves_existing_valid_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            override = Path(temp) / "override"
            for folder, file in ((base / "tcl/tcl8.6", "init.tcl"),
                                 (base / "tcl/tk8.6", "tk.tcl"),
                                 (override / "tcl", "init.tcl"),
                                 (override / "tk", "tk.tcl")):
                folder.mkdir(parents=True, exist_ok=True)
                (folder / file).write_text("", encoding="utf-8")
            env = {"TCL_LIBRARY": str(override / "tcl"), "TK_LIBRARY": str(override / "tk")}
            result = configure_tcl_tk_environment(base_prefix=base, environ=env)
            self.assertEqual(result["TCL_LIBRARY"], str(override / "tcl"))
            self.assertEqual(result["TK_LIBRARY"], str(override / "tk"))

    def test_upgrades_normal_base_path_when_tcl_cannot_traverse_junction(self):
        if os.name != "nt":
            self.skipTest("Windows extended-length base path")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            tcl = base / "tcl/tcl8.6"
            tk = base / "tcl/tk8.6"
            tcl.mkdir(parents=True); tk.mkdir(parents=True)
            (tcl / "init.tcl").write_text("", encoding="utf-8")
            (tk / "tk.tcl").write_text("", encoding="utf-8")
            env = {"TCL_LIBRARY": str(tcl), "TK_LIBRARY": str(tk)}
            result = configure_tcl_tk_environment(base_prefix=base, environ=env)
            self.assertTrue(result["TCL_LIBRARY"].startswith("\\\\?\\"))
            self.assertTrue(result["TK_LIBRARY"].startswith("\\\\?\\"))

    @unittest.skipUnless(os.name == "nt", "Windows native Tcl/Tk runtime check")
    def test_base_prefix_extended_path_opens_native_tk(self):
        configured = configure_tcl_tk_environment()
        import tkinter
        root = tkinter.Tk()
        try:
            self.assertTrue(root.tk.call("info", "patchlevel"))
            self.assertIn("TCL_LIBRARY", configured)
            self.assertIn("TK_LIBRARY", configured)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
