"""Native Tk replay smoke using the real, hash-verified D-082 predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tk_runtime import configure_tcl_tk_environment

configured = configure_tcl_tk_environment()
import tkinter as tk

from scripts.taco_stage5_material_hierarchy import VIEWS
from scripts.taco_stage5_display_profiles import NMS_CHOICES, THRESHOLD_CHOICES
from scripts.taco_stage5_visual_review import (
    CONSENSUS_STATE, ORGANIC_10_STATE, ORGANIC_9_STATE, ORGANIC_SHOWCASE_FILTER,
    Stage5ViewerGUI, discover_tier2_run, load_review_data,
)

run1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
highres = ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968"
tier1 = ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337"
tier2 = discover_tier2_run(root=ROOT)
assert tier2 is not None
dataset = ROOT / "exports/taco/strict-clean-stage5-v1"
data = load_review_data(run1, dataset, highres, tier1, tier2)
showcase = data["organic_showcase"]
assert showcase is not None, "Completed D-082 showcase artifact was not discovered"
assert showcase["manifest"]["detector_forwards"] == 8
for state in (ORGANIC_9_STATE, ORGANIC_10_STATE):
    assert set(data["predictions"][state]) == {68, 72, 92, 841}
    assert all(len(data["predictions"][state][image_id]) == 900 for image_id in (68, 72, 92, 841))

app = Stage5ViewerGUI(data, initial_image_id=68, default_threshold=.01, default_top_k=15)
root = app.root
root.withdraw()
root.update()
assert app.filter_var.get() == ORGANIC_SHOWCASE_FILTER
assert app.active_ids == [68, 72, 92, 841]
assert app.review_state_var.get() == ORGANIC_10_STATE

# Run every taxonomy/NMS/threshold combination on Image 68, then verify the
# saved 9-prompt, 10-prompt, and selected fourth-panel predictions on each image.
rendered = 0
for image_index, image_id in enumerate((68, 72, 92, 841)):
    app.set_current_image(image_id)
    combinations = (((view, mode, profile) for view in VIEWS for mode in NMS_CHOICES
                     for profile in THRESHOLD_CHOICES) if image_index == 0 else
                    ((view, app.nms_var.get(), app.threshold_mode_var.get()) for view in VIEWS))
    for taxonomy, mode, profile in combinations:
        app.taxonomy_var.set(taxonomy)
        app.nms_var.set(mode)
        app.threshold_mode_var.set(profile)
        app.update_plots()
        root.update()
        assert app.current_image_id == image_id
        assert mode in app.status_var.get() and profile in app.status_var.get()
        assert all(ax.images for ax in app.axes.values())
        if taxonomy == VIEWS[0]:
            assert any(child.cget("text") == "Organic"
                       for child in app.legend_frame.winfo_children())
        rendered += 1
    for state in (ORGANIC_9_STATE, ORGANIC_10_STATE):
        app.review_state_var.set(state)
        app.update_plots()
        root.update()
        table = app.tables["Stage 5 High-Res (384/576)"]
        assert len(table.get_children()) > 0, f"No saved {state} rows drawn for Image {image_id}"
    app.review_state_var.set(ORGANIC_10_STATE)

# Confirm the ordinary frozen 65-image view is unchanged and Image 777 still
# loads with the original validation state.
app.filter_var.set("All Images (65)")
app.populate_image_list()
app.on_filter_change()
app.set_current_image(777)
assert app.current_image_id == 777
assert app.review_state_var.get() == CONSENSUS_STATE
root.update()
assert all(ax.images for ax in app.axes.values())

for image_id in (68, 72, 92, 841):
    entry = showcase["images"][str(image_id)]
    photo = Path(showcase["dataset_root"]) / entry["file_name"]
    assert hashlib.sha256(photo.read_bytes()).hexdigest() == entry["file_sha256"]
root.destroy()
print(f"Native Tk saved Organic replay passed: {rendered} showcase control renders, "
      "9/10 prompt rows on all four images, plus frozen Image 777; "
      f"Tcl/Tk libraries={configured}.")
