"""Native Windows Tk smoke for the Stage 5 Image 777 diagnostic controls.

Usage: .venv-training/Scripts/python.exe -B tests/windows_stage5_image777_check.py
Opens only the frozen validation image and saved predictions; no model path is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tk_runtime import configure_tcl_tk_environment

configured = configure_tcl_tk_environment()
import tkinter as tk

from scripts.taco_stage5_display_profiles import NMS_CHOICES, THRESHOLD_CHOICES
from scripts.taco_stage5_material_hierarchy import VIEWS
from scripts.taco_stage5_visual_review import (
    DEFAULT_TIER1_RUN, discover_tier2_run, load_review_data, Stage5ViewerGUI,
)

run1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
highres = ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968"
tier1 = ROOT / DEFAULT_TIER1_RUN
tier2 = discover_tier2_run(root=ROOT)
if tier2 is None:
    raise SystemExit("Complete local Tier2 replay not found")
dataset = ROOT / "exports/taco/strict-clean-stage5-v1"
data = load_review_data(run1, dataset, highres, tier1, tier2)
photo = (data["dataset_root"] / data["images"][777]["file_name"]).resolve()
digest = hashlib.sha256(photo.read_bytes()).hexdigest()
app = Stage5ViewerGUI(data, initial_image_id=777, default_threshold=.10, default_top_k=15)
root = app.root
root.withdraw()
root.update()
assert app.current_image_id == 777
assert app.nms_var.get() in NMS_CHOICES
assert app.threshold_mode_var.get() == THRESHOLD_CHOICES[0]
assert set(app.preds) >= {"Tier 1 High-Res (384/576)", "Tier 2 E1 (384/576)"}

app.review_state_var.set("Tier 2 E1 (384/576)")
app.taxonomy_var.set(VIEWS[2])
app.nms_var.set("raw")
app.threshold_mode_var.set(THRESHOLD_CHOICES[0])
app.update_plots()
root.update()
table = app.tables["Stage 5 High-Res (384/576)"]
uniform_queries = {table.item(item, "values")[0] for item in table.get_children()}
assert "q58" not in uniform_queries, "0.070 cigarette should be hidden by uniform 0.10"

rendered = 0
for taxonomy in VIEWS:
    app.taxonomy_var.set(taxonomy)
    for mode in NMS_CHOICES:
        for profile in THRESHOLD_CHOICES:
            app.nms_var.set(mode)
            app.threshold_mode_var.set(profile)
            app.update_plots()
            root.update()
            assert taxonomy == app.taxonomy_var.get()
            assert mode in app.status_var.get()
            assert profile in app.status_var.get()
            assert all(ax.images for ax in app.axes.values())
            rendered += 1
            if taxonomy == VIEWS[2] and mode == "raw" and profile == THRESHOLD_CHOICES[1]:
                calibrated_queries = {table.item(item, "values")[0] for item in table.get_children()}
                assert "q58" in calibrated_queries, "0.070 cigarette should pass calibrated 0.06 floor"

assert hashlib.sha256(photo.read_bytes()).hexdigest() == digest
root.destroy()
print(f"Native Tk Image 777 passed: {rendered} taxonomy/NMS/threshold combinations; "
      f"Tcl/Tk libraries={configured}; Tier2 saved state, source photo hash unchanged.")
