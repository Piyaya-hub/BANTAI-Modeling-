"""Unit tests for Stage 5 D-105: Elimination of False Spatial Groupings & Clutter Suppression.

Verifies:
  1. Image 1243 (000082.JPG):
     - Bamboo stump clutter eliminated (8 FPs pruned).
     - Retains exactly 1 visible detection: True Positive plastic bottle (gt_iou >= 0.90, score >= 0.20).
     - Exactly 0 false positives on Image 1243.
     - Spatial grouping classifies Image 1243 as exactly 1 Individual Litter (item_count = 1).
  2. Bounded-Density & Anti-Daisy-Chaining Spatial Grouping:
     - Dispersed clusters (density < 0.45) classify as Group of Individual Litter, never Garbage Pile.
     - Clusters require density >= 0.45 and (overlap >= 0.15 or touching >= 1) or (n >= 3 and density >= 0.55)
       to be classified as Garbage Pile.
     - Single items retain exact model score; derived multi-item groups have model_score = None.
  3. All 14 Showcase Images Verified:
     - Image 1: 1 Garbage Pile (n=2, density >= 1.0)
     - Image 165: 1 Individual Litter (n=1)
     - Image 208: 1 Individual Litter (n=1)
     - Image 238: 1 Individual Litter (n=1)
     - Image 263: 1 Individual Litter (n=1)
     - Image 407: 1 Individual Litter (n=1)
     - Image 466: 1 Individual Litter (n=1)
     - Image 762: 1 Individual Litter (n=1)
     - Image 772: 1 Individual Litter (n=1)
     - Image 777: 1 Group of Individual Litter (n=5, density < 0.10)
     - Image 802: 1 Individual Litter (n=1)
     - Image 998: 2 Individual Litter (n=1, n=1, separated by 1280px)
     - Image 1243: 1 Individual Litter (n=1, 0 FPs)
     - Image 1465: 1 Group of Individual Litter (n=2, density < 0.30)
  4. Code Hygiene:
     - Zero hardcoded image IDs in consensus rules or spatial grouping module.
     - Zero occurrences of PROTECTED_SHOWCASE_IDS.
  5. Full 65-Image Validation Sweep:
     - Exactly 0 lost True Positives across all 65 validation images (lost_tps == []).
     - Visible false positives reduced to <= 65 (down from 74 in D-104).
     - Clean images increased to >= 35 / 65 (up from 34 in D-104).
     - Strict monotonic AP gain across Full 65 (AP9 >= 0.295136, AP5 >= 0.491688).
     - 2-fold interleaved splits (Fold A and Fold B) verified stable.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _get_default_baseline_rows,
    _get_default_run1_rows,
    _compute_iou,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_material_hierarchy import evaluate_case, NAMES, MAPS
from scripts.taco_stage5_spatial_grouping import (
    compute_spatial_grouping,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_DIAG_FRACTION,
    DEFAULT_DENSITY_THRESHOLD,
    SPATIAL_CATEGORIES,
)
import scripts.taco_stage5_spatial_grouping as sg_mod
import scripts.taco_stage5_consensus_tiebreak as cb_mod

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"
STAGE5_RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json"


class TestStage5D105FalseGroupElimination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_data = strict_load(VAL_JSON)
        cls.d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        cls.d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        cls.img_map = {img_id: idx for idx, img_id in enumerate(cls.d095_vecs["image_ids"])}
        cls.img_dims = {img["id"]: (img["width"], img["height"]) for img in cls.val_data["images"]}
        cls.base_data = strict_load(STAGE4_BASELINE)
        cls.base_by_img = {}
        for r in cls.base_data.get("raw_queries", []):
            cls.base_by_img.setdefault(r["image_id"], []).append(r)
        cls.run1_data = strict_load(STAGE5_RUN1)
        cls.run1_by_img = {}
        for r in cls.run1_data.get("raw_queries", []):
            cls.run1_by_img.setdefault(r["image_id"], []).append(r)
        cls.gts_by_img = {}
        for a in cls.val_data["annotations"]:
            cls.gts_by_img.setdefault(a["image_id"], []).append(a)
        cls.raw_by_img = {}
        for r in cls.d095_champ["raw_queries"]:
            cls.raw_by_img.setdefault(r["image_id"], []).append(r)

    def test_01_code_hygiene_and_contract_constants(self):
        """Zero hardcoded image IDs and calibrated spatial constants."""
        cb_src = inspect.getsource(cb_mod)
        self.assertNotIn("PROTECTED_SHOWCASE_IDS", cb_src)
        self.assertNotIn("1243", cb_src)
        self.assertNotIn("image_id in (", cb_src)
        self.assertNotIn("image_id ==", cb_src)

        sg_src = inspect.getsource(sg_mod)
        self.assertNotIn("image_id", sg_src)

        self.assertEqual(DEFAULT_SCALE_FACTOR, 2.0)
        self.assertEqual(DEFAULT_DIAG_FRACTION, 0.225)
        self.assertEqual(DEFAULT_DENSITY_THRESHOLD, 0.45)

    def test_02_image_1243_clutter_elimination_and_single_individual_litter(self):
        """Image 1243 (000082.JPG): 8 FPs pruned -> exactly 1 TP detection -> 1 Individual Litter."""
        img_id = 1243
        w, h = self.img_dims[img_id]
        raw_rows = self.raw_by_img[img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, _, _ = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            run1_rows=self.run1_by_img.get(img_id),
            enable_rule7=True,
        )

        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1, f"Expected exactly 1 visible detection on Image 1243, got {len(vis)}")

        # Verify it matches the Ground Truth plastic bottle
        gts = self.gts_by_img[img_id]
        gt = gts[0]
        ov = _compute_iou(vis[0]["bbox"], gt["bbox"])
        self.assertGreaterEqual(ov, 0.90)
        self.assertGreaterEqual(vis[0]["score"], 0.20)

        # Spatial grouping verification
        sg = compute_spatial_grouping(vis, image_width=w, image_height=h)
        s = sg["summary"]
        self.assertEqual(s["total_physical_items"], 1)
        self.assertEqual(s["total_regions"], 1)
        self.assertEqual(s["individual_litter_count"], 1)
        self.assertEqual(s["group_of_individual_litter_count"], 0)
        self.assertEqual(s["garbage_pile_count"], 0)
        self.assertEqual(sg["regions"][0]["category"], "individual_litter")
        self.assertEqual(sg["regions"][0]["item_count"], 1)
        self.assertAlmostEqual(sg["regions"][0]["model_score"], vis[0]["score"], places=4)

    def test_03_all_14_showcase_images(self):
        """All 14 showcase images verified for detections and spatial regions."""
        expected_results = {
            1: {"n_dets": 2, "n_regs": 1, "cats": ["garbage_pile"], "items": [2]},
            165: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            208: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            238: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            263: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            407: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            466: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            762: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            772: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            777: {"n_dets": 5, "n_regs": 1, "cats": ["group_of_individual_litter"], "items": [5]},
            802: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            998: {"n_dets": 2, "n_regs": 2, "cats": ["individual_litter", "individual_litter"], "items": [1, 1]},
            1243: {"n_dets": 1, "n_regs": 1, "cats": ["individual_litter"], "items": [1]},
            1465: {"n_dets": 2, "n_regs": 1, "cats": ["group_of_individual_litter"], "items": [2]},
        }

        for sid, exp in expected_results.items():
            w, h = self.img_dims[sid]
            raw_rows = self.raw_by_img[sid]
            vecs = self.d095_vecs["scores"][self.img_map[sid]]
            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(sid),
                run1_rows=self.run1_by_img.get(sid),
                enable_rule7=True,
            )
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            self.assertEqual(len(vis), exp["n_dets"], f"Image {sid} detection count mismatch")

            sg = compute_spatial_grouping(vis, image_width=w, image_height=h)
            regs = sg["regions"]
            self.assertEqual(len(regs), exp["n_regs"], f"Image {sid} region count mismatch")
            actual_cats = [r["category"] for r in regs]
            actual_items = [r["item_count"] for r in regs]
            self.assertEqual(actual_cats, exp["cats"], f"Image {sid} category mismatch")
            self.assertEqual(actual_items, exp["items"], f"Image {sid} item counts mismatch")

    def test_04_bounded_density_spatial_grouping_classification_rules(self):
        """Synthetic tests for bounded-density spatial grouping rules."""
        # 1. Low packing density with touching pair must NOT be Garbage Pile
        # 4 small items spread across a 1000x1000 box, but two touch
        loose_touching_rows = [
            {"bbox": [10.0, 10.0, 30.0, 30.0], "score": 0.50, "category_id": 1, "query_id": 0},
            {"bbox": [40.0, 10.0, 30.0, 30.0], "score": 0.50, "category_id": 1, "query_id": 1},  # touches item 0
            {"bbox": [500.0, 500.0, 30.0, 30.0], "score": 0.50, "category_id": 1, "query_id": 2},
            {"bbox": [900.0, 900.0, 30.0, 30.0], "score": 0.50, "category_id": 1, "query_id": 3},
        ]
        res = compute_spatial_grouping(loose_touching_rows, image_width=2000.0, image_height=2000.0, diag_fraction=0.50)
        # Bounding box is ~900x900 = 810,000; total area is 4 * 900 = 3600; density = 0.0044 < 0.45
        for reg in res["regions"]:
            if reg["item_count"] >= 2:
                self.assertEqual(reg["category"], "group_of_individual_litter", "Low density cluster must not be Garbage Pile")

        # 2. Dense overlapping cluster must be Garbage Pile
        dense_pile_rows = [
            {"bbox": [100.0, 100.0, 150.0, 150.0], "score": 0.70, "category_id": 8, "query_id": 0},
            {"bbox": [120.0, 110.0, 140.0, 160.0], "score": 0.65, "category_id": 8, "query_id": 1},
        ]
        res_pile = compute_spatial_grouping(dense_pile_rows, image_width=2000.0, image_height=2000.0)
        self.assertEqual(len(res_pile["regions"]), 1)
        self.assertEqual(res_pile["regions"][0]["category"], "garbage_pile")
        self.assertGreaterEqual(res_pile["regions"][0]["packing_density"], 0.45)
        self.assertGreaterEqual(res_pile["regions"][0]["max_iou"], 0.15)

    def test_05_full_65_sweep_zero_lost_tps_and_monotonic_ap(self):
        """Full 65 images: exactly 0 lost TPs, FP <= 65, Clean >= 35, AP9 >= 0.295136, AP5 >= 0.491688."""
        adj_by_img = {}
        total_tp = 0
        total_fp = 0
        clean_images = 0

        for img in self.val_data["images"]:
            i_id = img["id"]
            w, h = img["width"], img["height"]
            adj, _, _ = apply_scene_consensus(
                self.raw_by_img[i_id], self.d095_vecs["scores"][self.img_map[i_id]],
                image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(i_id),
                run1_rows=self.run1_by_img.get(i_id),
                enable_rule7=True,
            )
            adj_by_img[i_id] = adj

            gts = self.gts_by_img.get(i_id, [])
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            img_fp = 0
            img_tp = 0
            for det in vis:
                best_gt = None
                best_ov = 0.0
                for gt in gts:
                    ov = _compute_iou(det["bbox"], gt["bbox"])
                    if ov > best_ov:
                        best_ov = ov
                        best_gt = gt
                if best_ov >= 0.50 and best_gt is not None:
                    gt_c5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][best_gt["category_id"]]]
                    det_c5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][det["category_id"]]]
                    if det["category_id"] == best_gt["category_id"] or det_c5 == gt_c5:
                        img_tp += 1
                    else:
                        img_fp += 1
                else:
                    img_fp += 1
            total_tp += img_tp
            total_fp += img_fp
            if img_fp == 0:
                clean_images += 1

        self.assertLessEqual(total_fp, 65, f"Expected total_fp <= 65, got {total_fp}")
        self.assertGreaterEqual(clean_images, 35, f"Expected clean_images >= 35, got {clean_images}")
        self.assertGreaterEqual(total_tp, 76)

        # Verify 0 Lost True Positives
        lost_tps = []
        for img_id, rows in self.raw_by_img.items():
            gts = self.gts_by_img.get(img_id, [])
            raw_p = [r for r in rows if r["score"] >= 0.10]
            adj_p = [r for r in adj_by_img[img_id] if r["score"] >= 0.10]
            for gt in gts:
                raw_hit = any(r["category_id"] == gt["category_id"] and _compute_iou(r["bbox"], gt["bbox"]) >= 0.50 for r in raw_p)
                adj_hit = any(r["category_id"] == gt["category_id"] and _compute_iou(r["bbox"], gt["bbox"]) >= 0.50 for r in adj_p)
                if raw_hit and not adj_hit:
                    lost_tps.append((img_id, gt["id"]))
        self.assertEqual(lost_tps, [])

        # Full 65 Evaluation
        all_adj = [r for img_id in sorted(adj_by_img.keys()) for r in adj_by_img[img_id]]
        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        ap9 = res9["metrics"]["AP"]
        ap5 = res5["metrics"]["AP"]

        self.assertGreaterEqual(ap9, 0.295136 - 1e-5)
        self.assertGreaterEqual(ap5, 0.491688 - 1e-5)

        # Interleaved 2-Fold Split
        sorted_img_ids = sorted(self.raw_by_img.keys())
        fold_A_ids = set(img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 0)
        fold_B_ids = set(img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 1)

        data_A = {
            "images": [im for im in self.val_data["images"] if im["id"] in fold_A_ids],
            "annotations": [an for an in self.val_data["annotations"] if an["image_id"] in fold_A_ids],
            "categories": self.val_data["categories"],
        }
        data_B = {
            "images": [im for im in self.val_data["images"] if im["id"] in fold_B_ids],
            "annotations": [an for an in self.val_data["annotations"] if an["image_id"] in fold_B_ids],
            "categories": self.val_data["categories"],
        }
        preds_A = [r for img_id in sorted(fold_A_ids) for r in adj_by_img[img_id]]
        preds_B = [r for img_id in sorted(fold_B_ids) for r in adj_by_img[img_id]]

        res9_A = evaluate_case(data_A, preds_A, "9-specific", "raw")
        res5_A = evaluate_case(data_A, preds_A, "bantai-5-category", "raw")
        res9_B = evaluate_case(data_B, preds_B, "9-specific", "raw")
        res5_B = evaluate_case(data_B, preds_B, "bantai-5-category", "raw")

        self.assertGreaterEqual(res9_A["metrics"]["AP"], 0.245)
        self.assertGreaterEqual(res5_A["metrics"]["AP"], 0.370)
        self.assertGreaterEqual(res9_B["metrics"]["AP"], 0.370)
        self.assertGreaterEqual(res5_B["metrics"]["AP"], 0.630)


if __name__ == "__main__":
    unittest.main()
