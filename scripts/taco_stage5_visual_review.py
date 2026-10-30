#!/usr/bin/env python3
"""
Stage 5 Visual Review: Interactive Tk/TkAgg local viewer for 9-class waste detections.

Replays verified predictions from Stage 5 run across the 65 validation images.
Compares:
  - Ground Truth
  - Baseline (Warm-start)
  - Epoch 1
  - Epoch 2
Zero model execution or GPU memory needed during review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_training_labels_v2 import LABELS
from scripts.taco_stage5_material_hierarchy import (VIEWS, VIEW_SCHEMES, NAMES, COLORS, map_row, prepare_rows)

# 9 distinct high-contrast colors for waste categories
CLASS_COLORS = [
    "#e6194B",  # 0: cigarette (crimson red)
    "#3cb44b",  # 1: plastic_bottle (green)
    "#ffe119",  # 2: plastic_bag (yellow)
    "#4363d8",  # 3: plastic_wrapper (blue)
    "#f58231",  # 4: plastic_film (orange)
    "#911eb4",  # 5: drink_can (purple)
    "#42d4f4",  # 6: plastic_cup (cyan)
    "#f032e6",  # 7: styrofoam_piece (magenta)
    "#808000",  # 8: paper (olive gold / amber)
]


DEFAULT_TIER1_RUN = Path("runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337")


def discover_tier1_run(explicit=None, root=ROOT):
    """Explicit paths win; absent/incomplete private defaults fall back to old review."""
    if explicit is not None:
        return Path(explicit)
    candidate=Path(root)/DEFAULT_TIER1_RUN
    required=('report.json','validation-epoch-1.json','soup_tier1_hr_blend2575.json','soup_tier1_hr_5050.json')
    return candidate if all((candidate/name).is_file() for name in required) else None


def apply_containment_nms(objects, iou_thresh: float = 0.50, iomin_thresh: float = 0.80):
    """
    Suppresses duplicate bounding boxes using IoU and Containment (IoMin).
    If a smaller box is >=80% contained in a larger box, or if IoU >= 50%, suppress it.
    """
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda x: -x.get("score", 1.0))
    kept = []
    for obj in sorted_objs:
        bx, by, bw, bh = obj["bbox"]
        area = bw * bh
        if area <= 0:
            continue
        suppressed = False
        for k_obj in kept:
            if obj["category_id"] != k_obj["category_id"]:
                continue
            kx, ky, kw, kh = k_obj["bbox"]
            k_area = kw * kh
            if k_area <= 0:
                continue
            ix1 = max(bx, kx)
            iy1 = max(by, ky)
            ix2 = min(bx + bw, kx + kw)
            iy2 = min(by + bh, ky + kh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter > 0:
                union = area + k_area - inter
                iou = inter / union if union > 0 else 0
                iomin = inter / min(area, k_area) if min(area, k_area) > 0 else 0
                if iou >= iou_thresh or iomin >= iomin_thresh:
                    suppressed = True
                    break
        if not suppressed:
            kept.append(obj)
    return kept


def load_review_data(run_dir: Path, dataset_dir: Path, highres_run_dir: Path | None = None, tier1_run_dir: Path | None = None):
    val_json_path = dataset_dir / "validation.json"
    val_data = strict_load(val_json_path)

    images = {img["id"]: img for img in val_data["images"]}
    annotations_by_img = defaultdict(list)
    for ann in val_data["annotations"]:
        annotations_by_img[ann["image_id"]].append(ann)

    # Load prediction files from base run
    baseline_path = run_dir / "baseline.json"
    ep1_path = run_dir / "validation-epoch-1.json"
    ep2_path = run_dir / "validation-epoch-2.json"

    baseline_data = strict_load(baseline_path) if baseline_path.is_file() else {}
    ep1_data = strict_load(ep1_path) if ep1_path.is_file() else {}
    ep2_data = strict_load(ep2_path) if ep2_path.is_file() else {}

    # Check for high-res run
    highres_ep1_data = {}
    if highres_run_dir and (highres_run_dir / "validation-epoch-1.json").is_file():
        highres_ep1_data = strict_load(highres_run_dir / "validation-epoch-1.json")
    else:
        # Default high-res run location if available
        def_highres = ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968"
        if (def_highres / "validation-epoch-1.json").is_file():
            highres_ep1_data = strict_load(def_highres / "validation-epoch-1.json")

    preds_by_state = {
        "Baseline": defaultdict(list),
        "Stage 5 (256/384)": defaultdict(list),
        "Stage 5 High-Res (384/576)": defaultdict(list),
    }

    for p in baseline_data["raw_queries"]:
        preds_by_state["Baseline"][p["image_id"]].append(p)
    for p in ep2_data["raw_queries"]:
        preds_by_state["Stage 5 (256/384)"][p["image_id"]].append(p)
    for p in highres_ep1_data["raw_queries"]:
        preds_by_state["Stage 5 High-Res (384/576)"][p["image_id"]].append(p)

    if set(images) != {p["image_id"] for p in highres_ep1_data["raw_queries"]}:
        raise ValueError("High-resolution saved validation predictions are incomplete")

    if tier1_run_dir is not None:
        report = strict_load(tier1_run_dir / "report.json")
        if not report.get("completed") or report.get("training_updates") != 336:
            raise ValueError("Tier1 replay requires a completed bounded run")
        for label, filename in (
            ("Tier 1 High-Res (384/576)", "validation-epoch-1.json"),
            ("Tier 1 + blend2575 soup", "soup_tier1_hr_blend2575.json"),
            ("Tier 1 + HR E1 soup", "soup_tier1_hr_5050.json"),
        ):
            saved = strict_load(tier1_run_dir / filename)
            rows = saved['raw_queries']
            if len(rows) != 65*900 or {r['image_id'] for r in rows} != set(images):
                raise ValueError("Tier1 saved queries differ from frozen validation")
            preds_by_state[label] = defaultdict(list)
            for row in rows: preds_by_state[label][row['image_id']].append(row)

    # Sort predictions by score descending
    for state in preds_by_state:
        for img_id in preds_by_state[state]:
            preds_by_state[state][img_id].sort(key=lambda x: -x["score"])

    return {
        "images": images,
        "dataset_root": Path(val_data["info"]["image_root"]).resolve(),
        "ground_truth": annotations_by_img,
        "predictions": preds_by_state,
        "metrics": {
            "Baseline": baseline_data.get("combined_65", {}),
            "Stage 5 (256/384)": ep2_data.get("combined_65", {}),
            "Stage 5 High-Res (384/576)": highres_ep1_data.get("combined_65", {}),
        },
    }


def draw_panel(ax, image, objects, is_gt: bool = False, score_thresh: float = 0.10, top_k: int = 15):
    from matplotlib.patches import Rectangle

    w, h = image.size
    ax.clear()
    ax.imshow(image)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    rendered = 0
    for obj in objects:
        if not is_gt:
            if obj.get("score", 1.0) < score_thresh:
                continue
            if rendered >= top_k:
                break

        cid = obj["category_id"]
        color = CLASS_COLORS[cid] if 0 <= cid < len(CLASS_COLORS) else "#ffffff"
        label_name = obj.get("display_label", LABELS[cid] if 0 <= cid < len(LABELS) else f"cls_{cid}")
        color = COLORS.get(obj.get("mapped_category"), color)

        bx, by, bw, bh = obj["bbox"]
        linestyle = "-" if is_gt else "--"
        linewidth = 1.8 if is_gt else 1.5

        rect = Rectangle(
            (bx, by), bw, bh,
            fill=False, edgecolor=color,
            linestyle=linestyle, linewidth=linewidth,
            zorder=3,
        )
        ax.add_patch(rect)

        # Label badge
        score_str = f" {obj['score']:.2f}" if not is_gt else ""
        text = f"{label_name}{score_str}"
        ax.text(
            bx, max(0, by - 4), text,
            fontsize=7, weight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            zorder=4,
        )
        rendered += 1


class Stage5ViewerGUI:
    def __init__(self, data: dict, initial_image_id: int | None = None, default_threshold: float = 0.10, default_top_k: int = 15):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.data = data
        self.images = data["images"]
        self.gt = data["ground_truth"]
        self.preds = data["predictions"]
        self.dataset_root = data["dataset_root"]

        # Identify which images have paper
        self.paper_img_ids = set()
        for img_id, anns in self.gt.items():
            if any(a["category_id"] == 8 for a in anns):
                self.paper_img_ids.add(img_id)

        self.root = tk.Tk()
        self.root.title("BANTAI — Stage 5 9-Class Waste Detection Visual Review")
        self.root.geometry("1480x920")

        # Top Control Bar
        ctrl_frame = ttk.Frame(self.root, padding=6)
        ctrl_frame.pack(side="top", fill="x")

        # Filter mode
        ttk.Label(ctrl_frame, text="Filter:").pack(side="left", padx=(4, 2))
        self.filter_var = tk.StringVar(value="Paper Images (18)" if self.paper_img_ids else "All Images (65)")
        filter_cb = ttk.Combobox(ctrl_frame, textvariable=self.filter_var, values=["Paper Images (18)", "All Images (65)"], state="readonly", width=16)
        filter_cb.pack(side="left", padx=4)
        filter_cb.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Image Dropdown
        ttk.Label(ctrl_frame, text="Image:").pack(side="left", padx=(10, 2))
        self.img_var = tk.StringVar()
        self.img_cb = ttk.Combobox(ctrl_frame, textvariable=self.img_var, state="readonly", width=26)
        self.img_cb.pack(side="left", padx=4)
        self.img_cb.bind("<<ComboboxSelected>>", self.on_image_change)

        # Prev / Next Buttons
        ttk.Button(ctrl_frame, text="◀ Prev", width=7, command=self.prev_image).pack(side="left", padx=2)
        ttk.Button(ctrl_frame, text="Next ▶", width=7, command=self.next_image).pack(side="left", padx=2)

        # Threshold Control
        ttk.Label(ctrl_frame, text="Score Threshold:").pack(side="left", padx=(14, 2))
        self.thresh_var = tk.DoubleVar(value=default_threshold)
        thresh_spin = ttk.Spinbox(ctrl_frame, from_=0.01, to=0.99, increment=0.02, textvariable=self.thresh_var, width=6, command=self.update_plots)
        thresh_spin.pack(side="left", padx=2)
        thresh_spin.bind("<Return>", lambda e: self.update_plots())

        # Top-K Control
        ttk.Label(ctrl_frame, text="Top-K:").pack(side="left", padx=(10, 2))
        self.topk_var = tk.IntVar(value=default_top_k)
        topk_spin = ttk.Spinbox(ctrl_frame, from_=1, to=100, increment=5, textvariable=self.topk_var, width=5, command=self.update_plots)
        topk_spin.pack(side="left", padx=2)
        topk_spin.bind("<Return>", lambda e: self.update_plots())

        self.nms_var = tk.StringVar(value='class_aware_iou_0.50_containment_0.80')
        nms_cb = ttk.Combobox(ctrl_frame, textvariable=self.nms_var,
            values=('raw','class_aware_iou_0.50','class_aware_iou_0.50_containment_0.80'),
            state="readonly", width=24)
        nms_cb.pack(side="left", padx=4)
        nms_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        ttk.Label(ctrl_frame, text="Taxonomy View:").pack(side="left", padx=4)
        self.taxonomy_var = tk.StringVar(value=VIEWS[2])
        taxonomy_cb = ttk.Combobox(ctrl_frame, textvariable=self.taxonomy_var,
                                  values=VIEWS, state="readonly", width=38)
        taxonomy_cb.pack(side="left")
        taxonomy_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        state_frame = ttk.Frame(self.root)
        state_frame.pack(side="top", fill="x")
        ttk.Label(state_frame, text="Fourth panel state:").pack(side="left", padx=6)
        choices=[n for n in self.preds if n not in ("Baseline", "Stage 5 (256/384)")]
        initial="Tier 1 + blend2575 soup" if "Tier 1 + blend2575 soup" in choices else choices[0]
        self.review_state_var=tk.StringVar(value=initial)
        state_cb=ttk.Combobox(state_frame, textvariable=self.review_state_var, values=choices,
                              state="readonly", width=38)
        state_cb.pack(side="left")
        state_cb.bind("<<ComboboxSelected>>", lambda e:self.update_plots())
        self.legend_frame = ttk.Frame(self.root)
        self.legend_frame.pack(side="top", fill="x")

        # Main Paned Window (Plots on top, Details table on bottom, exactly like Stage 4)
        pane = ttk.Panedwindow(self.root, orient="vertical")
        pane.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(pane)
        details_frame = ttk.Frame(pane)
        pane.add(plot_frame, weight=3)
        pane.add(details_frame, weight=1)

        # Matplotlib Figure (2x2 Grid)
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.axes = {
            "GT": self.fig.add_subplot(2, 2, 1),
            "Baseline": self.fig.add_subplot(2, 2, 2),
            "Stage 5 (256/384)": self.fig.add_subplot(2, 2, 3),
            "Stage 5 High-Res (384/576)": self.fig.add_subplot(2, 2, 4),
        }
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Details Notebook (Tables for GT, Baseline, Stage 5 Run 1, Stage 5 High-Res)
        notebook = ttk.Notebook(details_frame)
        self.details_notebook = notebook
        notebook.pack(fill="both", expand=True)
        self.tables = {}
        columns = ("rank", "category", "specific_class", "hazard_flag", "score", "bbox_xywh")
        for tab_name in ("Ground Truth", "Baseline", "Stage 5 (256/384)", "Stage 5 High-Res (384/576)"):
            tab_f = ttk.Frame(notebook)
            notebook.add(tab_f, text=tab_name)
            tree = ttk.Treeview(tab_f, columns=columns, show="headings", height=5)
            tree.heading("rank", text="Rank")
            tree.heading("category", text="Category")
            tree.heading("specific_class", text="specific_class")
            tree.heading("hazard_flag", text="hazard_flag")
            tree.column("specific_class", width=150)
            tree.column("hazard_flag", width=80)
            tree.heading("score", text="Score")
            tree.heading("bbox_xywh", text="BBox [x, y, w, h]")
            tree.column("rank", width=60, stretch=False)
            tree.column("category", width=140, stretch=False)
            tree.column("score", width=100, stretch=False)
            tree.column("bbox_xywh", width=320, stretch=True)
            y_sb = ttk.Scrollbar(tab_f, orient="vertical", command=tree.yview)
            x_sb = ttk.Scrollbar(tab_f, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_sb.set, xscrollcommand=x_sb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_sb.grid(row=0, column=1, sticky="ns")
            x_sb.grid(row=1, column=0, sticky="ew")
            tab_f.rowconfigure(0, weight=1)
            tab_f.columnconfigure(0, weight=1)
            self.tables[tab_name] = tree

        # Status Bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=4)
        status_bar.pack(side="bottom", fill="x")

        self.populate_image_list()
        if initial_image_id is not None and initial_image_id in self.images:
            self.filter_var.set("All Images (65)")
            self.populate_image_list()
            self.set_current_image(initial_image_id)
        else:
            first_id = sorted(self.paper_img_ids)[0] if self.paper_img_ids else sorted(self.images.keys())[0]
            self.set_current_image(first_id)

    def populate_image_list(self):
        f = self.filter_var.get()
        if "Paper" in f:
            target_ids = sorted(self.paper_img_ids)
        else:
            target_ids = sorted(self.images.keys())

        items = []
        for img_id in target_ids:
            img = self.images[img_id]
            tag = " [PAPER]" if img_id in self.paper_img_ids else ""
            fname = Path(img["file_name"]).name
            items.append(f"Image {img_id}{tag} ({fname})")

        self.img_cb["values"] = items
        self.active_ids = target_ids

    def on_filter_change(self, event=None):
        self.populate_image_list()
        if self.active_ids:
            self.set_current_image(self.active_ids[0])

    def on_image_change(self, event=None):
        selection = self.img_var.get()
        if not selection:
            return
        img_id_str = selection.split()[1]
        img_id = int(img_id_str)
        self.load_and_draw(img_id)

    def set_current_image(self, img_id: int):
        for val in self.img_cb["values"]:
            if val.startswith(f"Image {img_id} "):
                self.img_var.set(val)
                self.load_and_draw(img_id)
                break

    def prev_image(self):
        if not self.active_ids:
            return
        cur_id = self.current_image_id
        if cur_id in self.active_ids:
            idx = self.active_ids.index(cur_id)
            new_id = self.active_ids[(idx - 1) % len(self.active_ids)]
            self.set_current_image(new_id)

    def next_image(self):
        if not self.active_ids:
            return
        cur_id = self.current_image_id
        if cur_id in self.active_ids:
            idx = self.active_ids.index(cur_id)
            new_id = self.active_ids[(idx + 1) % len(self.active_ids)]
            self.set_current_image(new_id)

    def load_and_draw(self, img_id: int):
        self.current_image_id = img_id
        img_entry = self.images[img_id]
        photo_path = (self.dataset_root / img_entry["file_name"]).resolve()
        if not photo_path.is_relative_to(self.dataset_root):
            raise ValueError("Validation photo path escapes dataset root")

        if not photo_path.is_file():
            self.status_var.set(f"ERROR: Photograph not found at {photo_path}")
            return

        import hashlib
        expected = img_entry["source_taco"]["file_sha256"]
        if hashlib.sha256(photo_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Validation source photo hash mismatch")
        with Image.open(photo_path) as pil_img:
            self.current_pil = pil_img.convert("RGB")

        self.update_plots()

    def update_plots(self):
        if not hasattr(self, "current_pil") or not hasattr(self, "current_image_id"):
            return

        img_id = self.current_image_id
        self.details_notebook.tab(3, text=self.review_state_var.get())
        thresh = float(self.thresh_var.get())
        top_k = int(self.topk_var.get())

        scheme = VIEW_SCHEMES[self.taxonomy_var.get()]
        hierarchical = self.taxonomy_var.get() == VIEWS[3]
        import tkinter as tk
        for child in self.legend_frame.winfo_children(): child.destroy()
        for name in NAMES[scheme]:
            tk.Label(self.legend_frame, text=name, bg=COLORS.get(name, CLASS_COLORS[LABELS.index(name)] if name in LABELS else "white"), fg="white").pack(side="left", padx=2)
        gt_boxes = [map_row(a, scheme, hierarchical) for a in self.gt.get(img_id, [])]
        base_preds = self.preds["Baseline"].get(img_id, [])
        run1_preds = self.preds["Stage 5 (256/384)"].get(img_id, [])
        highres_preds = self.preds[self.review_state_var.get()].get(img_id, [])

        # Apply score threshold and optional containment NMS
        def filter_preds(plist):
            mode = self.nms_var.get()
            return prepare_rows(plist, scheme, mode, thresh, hierarchical)[:top_k]

        base_filtered = filter_preds(base_preds)
        run1_filtered = filter_preds(run1_preds)
        highres_filtered = filter_preds(highres_preds)

        draw_panel(self.axes["GT"], self.current_pil, gt_boxes, is_gt=True)
        self.axes["GT"].set_title(f"Ground Truth ({len(gt_boxes)} objects)", fontsize=11, weight="bold")

        draw_panel(self.axes["Baseline"], self.current_pil, base_filtered, is_gt=False, score_thresh=thresh, top_k=top_k)
        self.axes["Baseline"].set_title(f"Warm-Start Baseline (@{thresh:.2f})", fontsize=11)

        draw_panel(self.axes["Stage 5 (256/384)"], self.current_pil, run1_filtered, is_gt=False, score_thresh=thresh, top_k=top_k)
        self.axes["Stage 5 (256/384)"].set_title(f"Stage 5 Run 1 (256/384; diagnostic view)", fontsize=11)

        draw_panel(self.axes["Stage 5 High-Res (384/576)"], self.current_pil, highres_filtered, is_gt=False, score_thresh=thresh, top_k=top_k)
        self.axes["Stage 5 High-Res (384/576)"].set_title(self.review_state_var.get(), fontsize=11, color="#005500", weight="bold")

        self.canvas.draw_idle()

        # Update Details Tables
        for tree in self.tables.values():
            tree.delete(*tree.get_children())

        # Ground Truth table
        for r_idx, ann in enumerate(gt_boxes, 1):
            c_name = ann["display_label"]
            bx = [round(v, 1) for v in ann["bbox"]]
            self.tables["Ground Truth"].insert("", "end", values=(f"GT-{ann['id']}", c_name, ann["specific_class"], ann["hazard_flag"], "1.000", str(bx)))

        # Baseline, Stage 5 Run 1, Stage 5 High-Res tables
        for tab_name, passing in [
            ("Baseline", base_filtered),
            ("Stage 5 (256/384)", run1_filtered),
            ("Stage 5 High-Res (384/576)", highres_filtered),
        ]:
            for r_idx, p in enumerate(passing, 1):
                c_name = p["display_label"]
                bx = [round(v, 1) for v in p["bbox"]]
                self.tables[tab_name].insert("", "end", values=(f"q{p['query_id']}", c_name, p["specific_class"], p["hazard_flag"], f"{p['score']:.4f}", str(bx)))

        # Update status bar
        gt_types = [a["display_label"] for a in gt_boxes]
        hr_types = [f"{p['display_label']}({p['score']:.2f})" for p in highres_filtered]
        nms_str = " | NMS: " + self.nms_var.get()
        self.status_var.set(
            f"Image ID: {img_id} [{self.current_pil.size[0]}x{self.current_pil.size[1]}]{nms_str} | "
            f"GT: {len(gt_boxes)} {gt_types} | "
            f"High-Res Detections (>= {thresh:.2f}, top-{top_k}): {len(highres_filtered)} {hr_types}"
        )

    def run(self):
        self.root.mainloop()


def print_cli_summary(data: dict, score_thresh: float, top_k: int, image_id: int | None = None):
    print("=== Stage 5 Visual Review Summary ===")
    target_ids = [image_id] if image_id else sorted(data["images"].keys())

    for img_id in target_ids:
        img = data["images"][img_id]
        gt_list = data["ground_truth"].get(img_id, [])
        gt_names = [LABELS[a["category_id"]] for a in gt_list]

        run1_preds = [p for p in data["predictions"]["Stage 5 (256/384)"].get(img_id, []) if p["score"] >= score_thresh][:top_k]
        hr_preds = [p for p in data["predictions"]["Stage 5 High-Res (384/576)"].get(img_id, []) if p["score"] >= score_thresh][:top_k]

        pred1_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in run1_preds]
        hr_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in hr_preds]

        paper_mark = " [PAPER]" if any(a["category_id"] == 8 for a in gt_list) else ""
        print(f"\nImage {img_id}{paper_mark} ({img['file_name']}) [{img['width']}x{img['height']}]:")
        print(f"  Ground Truth ({len(gt_list)}): {', '.join(gt_names) if gt_names else 'None'}")
        print(f"  Stage 5 Run 1  @ {score_thresh:.2f} ({len(run1_preds)}): {', '.join(pred1_desc) if pred1_desc else 'None'}")
        print(f"  Stage 5 HighRes @ {score_thresh:.2f} ({len(hr_preds)}): {', '.join(hr_desc) if hr_desc else 'None'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 Nine-Class Waste Detection Visual Review")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4",
        help="Path to Stage 5 Run 1 training directory",
    )
    parser.add_argument(
        "--highres-run-dir",
        type=Path,
        default=ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968",
        help="Path to Stage 5 High-Res training directory",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "exports/taco/strict-clean-stage5-v1",
        help="Path to Stage 5 dataset export",
    )
    parser.add_argument("--tier1-run-dir", type=Path, default=None, help="Explicit Tier1 replay directory; auto-discovers the completed local D074 run by default")
    parser.add_argument("--gui", action="store_true", help="Launch interactive TkAgg GUI viewer")
    parser.add_argument("--image-id", type=int, default=None, help="Inspect specific Image ID")
    parser.add_argument("--score-threshold", type=float, default=0.10, help="Confidence score threshold (default: 0.10)")
    parser.add_argument("--box-top-k", type=int, default=15, help="Maximum boxes drawn per panel (default: 15)")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"ERROR: Run directory not found: {args.run_dir}", file=sys.stderr)
        return 1

    args.tier1_run_dir = discover_tier1_run(args.tier1_run_dir)
    if args.tier1_run_dir is not None and not args.tier1_run_dir.is_dir():
        print(f"ERROR: Explicit Tier1 saved replay directory is missing: {args.tier1_run_dir}", file=sys.stderr)
        return 1
    if not (args.dataset_dir / "validation.json").is_file():
        print("ERROR: Saved review needs the private validation export; no model will be run.", file=sys.stderr)
        return 1
    try:
        data = load_review_data(args.run_dir, args.dataset_dir, args.highres_run_dir, args.tier1_run_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: Saved replay artifacts are unavailable/incomplete ({type(exc).__name__}); supply local run/export paths. No model will be run.", file=sys.stderr)
        return 1

    if args.gui:
        app = Stage5ViewerGUI(
            data,
            initial_image_id=args.image_id,
            default_threshold=args.score_threshold,
            default_top_k=args.box_top_k,
        )
        app.run()
    else:
        print_cli_summary(data, args.score_threshold, args.box_top_k, args.image_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
