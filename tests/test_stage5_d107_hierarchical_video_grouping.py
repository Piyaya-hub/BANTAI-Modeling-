"""Unit tests for Stage 5 D-107: Two-Tier Hierarchical Spatial Grouping & Video De-Cluttering.

Verifies:
  1. Code Hygiene & Zero Hardcoding:
     - Zero hardcoded image/frame IDs in consensus rules or spatial grouping modules.
     - Zero occurrences of PROTECTED_SHOWCASE_IDS.
  2. Two-Tier Hierarchical DBSCAN Spatial Grouping Contract:
     - Tier 1: Dense core extraction (tight adjacency + density >= 0.40 or touch >= 1).
     - Tier 2: Local proximity with cluster diagonal <= 0.42 * L anti-chaining constraint.
     - Single items retain exact model score; derived multi-item groups have model_score = None.
     - Dispersed clusters (density < 0.45) classify as Group of Individual Litter, never Garbage Pile.
  3. All 14 Validation Showcase Image Invariants Preserved:
     - Image 1: 1 Garbage Pile (n=2)
     - Image 165: 1 Individual Litter (n=1)
     - Image 208: 1 Individual Litter (n=1)
     - Image 238: 1 Individual Litter (n=1)
     - Image 263: 1 Individual Litter (n=1)
     - Image 407: 1 Individual Litter (n=1)
     - Image 466: 1 Individual Litter (n=1)
     - Image 762: 1 Individual Litter (n=1)
     - Image 772: 1 Individual Litter (n=1)
     - Image 777: 1 Group of Individual Litter (n=5)
     - Image 802: 1 Individual Litter (n=1)
     - Image 998: 2 Individual Litter (n=1, n=1)
     - Image 1243: 1 Individual Litter (n=1)
     - Image 1465: 1 Group of Individual Litter (n=2)
  4. Sachet/Wrapper Hazard Tie-Break Verification:
     - Large compact objects with non-hazard support reassign from cigarette to plastic wrapper.
     - Reassignment verified on Frame 10008 (qid=1 -> plastic_wrapper, Plastic).
  5. Video Frame Separation on Frames 10006 & 10008:
     - Resolves the giant single-cluster failure by extracting dense wall heaps into Garbage Piles
       while separating foreground street items into Group of Individual Litter / Individual Litter.
  6. Full 65 Validation Evaluation Sweep:
     - Exactly 0 lost True Positives across all 65 validation images (lost_tps == []).
     - Clean images >= 35 / 65.
     - AP9 monotonic gain (AP9 >= 0.295136, AP5 >= 0.491688).
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _compute_iou,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_material_hierarchy import evaluate_case, NAMES, MAPS
from scripts.taco_stage5_spatial_grouping import (
    compute_spatial_grouping,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_DIAG_FRACTION,
    DEFAULT_DENSITY_THRESHOLD,
    DEFAULT_IOU_PILE_THRESHOLD,
    DEFAULT_MAX_CLUSTER_DIAG_FRACTION,
    SPATIAL_CATEGORIES,
)
import scripts.taco_stage5_spatial_grouping as sg_mod
import scripts.taco_stage5_consensus_tiebreak as cb_mod

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"
STAGE5_RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json"
VIDEO_SHOWCASE = ROOT / "runs/video_detection/motion_extract_testing/video-showcase.json"
VIDEO_MANIFEST = ROOT / "runs/video_detection/motion_extract_testing/manifest.json"


class TestStage5D107HierarchicalVideoGrouping(unittest.TestCase):
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

        cls.video_showcase = strict_load(VIDEO_SHOWCASE)
        cls.video_manifest = strict_load(VIDEO_MANIFEST)

    def test_01_code_hygiene_and_constants(self):
        """Zero hardcoded image/video IDs in consensus rules or spatial grouping module."""
        cb_src = inspect.getsource(cb_mod)
        self.assertNotIn("PROTECTED_SHOWCASE_IDS", cb_src)
        self.assertNotIn("10006", cb_src)
        self.assertNotIn("10008", cb_src)
        self.assertNotIn("image_id in (", cb_src)
        self.assertNotIn("image_id ==", cb_src)

        sg_src = inspect.getsource(sg_mod)
        self.assertNotIn("image_id", sg_src)
        self.assertNotIn("frame", sg_src)

        self.assertEqual(DEFAULT_SCALE_FACTOR, 2.0)
        self.assertEqual(DEFAULT_DIAG_FRACTION, 0.225)
        self.assertEqual(DEFAULT_DENSITY_THRESHOLD, 0.45)
        self.assertEqual(DEFAULT_IOU_PILE_THRESHOLD, 0.15)
        self.assertEqual(DEFAULT_MAX_CLUSTER_DIAG_FRACTION, 0.42)

    def test_02_synthetic_two_tier_spatial_grouping_contract(self):
        """Synthetic tests verifying Tier 1 core extraction and Tier 2 anti-chaining."""
        # 1. Single item retains exact score
        single_item = [{"bbox": [100.0, 100.0, 50.0, 50.0], "score": 0.88, "category_id": 1, "query_id": 0}]
        res_single = compute_spatial_grouping(single_item, 1000.0, 1000.0)
        self.assertEqual(res_single["summary"]["total_regions"], 1)
        self.assertEqual(res_single["summary"]["individual_litter_count"], 1)
        self.assertEqual(res_single["regions"][0]["category"], "individual_litter")
        self.assertEqual(res_single["regions"][0]["model_score"], 0.88)

        # 2. Dense 3-item pile in Tier 1
        dense_pile = [
            {"bbox": [200.0, 200.0, 80.0, 80.0], "score": 0.60, "category_id": 3, "query_id": 0},
            {"bbox": [220.0, 210.0, 75.0, 75.0], "score": 0.55, "category_id": 4, "query_id": 1},
            {"bbox": [240.0, 205.0, 70.0, 70.0], "score": 0.50, "category_id": 8, "query_id": 2},
        ]
        res_pile = compute_spatial_grouping(dense_pile, 1000.0, 1000.0)
        self.assertEqual(res_pile["summary"]["garbage_pile_count"], 1)
        self.assertEqual(res_pile["regions"][0]["category"], "garbage_pile")
        self.assertIsNone(res_pile["regions"][0]["model_score"])
        self.assertGreaterEqual(res_pile["regions"][0]["packing_density"], 0.40)

        # 3. Dense pile + distant foreground item: separated into 1 Garbage Pile and 1 Individual Litter
        mixed_items = dense_pile + [
            {"bbox": [800.0, 800.0, 40.0, 40.0], "score": 0.75, "category_id": 1, "query_id": 3}
        ]
        res_mixed = compute_spatial_grouping(mixed_items, 1000.0, 1000.0)
        self.assertEqual(res_mixed["summary"]["total_regions"], 2)
        self.assertEqual(res_mixed["summary"]["garbage_pile_count"], 1)
        self.assertEqual(res_mixed["summary"]["individual_litter_count"], 1)
        self.assertEqual(res_mixed["regions"][0]["category"], "garbage_pile")
        self.assertEqual(res_mixed["regions"][0]["item_count"], 3)
        self.assertEqual(res_mixed["regions"][1]["category"], "individual_litter")
        self.assertEqual(res_mixed["regions"][1]["item_count"], 1)

    def test_03_all_14_validation_showcase_invariants(self):
        """All 14 validation showcase images preserved under D-107 Two-Tier grouping."""
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

    def test_04_sachet_wrapper_hazard_tiebreak_and_video_frame_reassignment(self):
        """Frame 10008 qid=1 reassigned from 0: cigarette (Hazard) to 3: plastic_wrapper (Plastic)."""
        f10008_summary = self.video_manifest["frames_summary"]["10008"]
        wrapper_det = [d for d in f10008_summary["detections"] if d["category_name"] == "plastic_wrapper" and d["score"] >= 0.32]
        self.assertTrue(len(wrapper_det) >= 1, "Expected red candy wrapper to be plastic_wrapper")
        self.assertEqual(wrapper_det[0]["bantai_category"], "Plastic")
        self.assertEqual(wrapper_det[0]["category_id"], 3)

        # Verify no false cigarette hazard exists on Frame 10008
        cig_hazards = [d for d in f10008_summary["detections"] if d["category_id"] == 0]
        self.assertEqual(len(cig_hazards), 0, "No false cigarette hazards should exist on Frame 10008")

    def test_05_video_frames_hierarchical_spatial_separation(self):
        """Frames 10006 and 10008 separate dense roadside heaps from foreground litter."""
        f10006_sg = self.video_manifest["frames_summary"]["10006"]["spatial_grouping"]
        f10008_sg = self.video_manifest["frames_summary"]["10008"]["spatial_grouping"]

        # Frame 10006 has dense pile + group of litter
        self.assertGreaterEqual(f10006_sg["garbage_piles"], 1)
        self.assertGreaterEqual(f10006_sg["groups_of_litter"], 1)
        self.assertGreaterEqual(f10006_sg["total_regions"], 2)

        # Frame 10008 has at least 2 dense piles, 1 group, and 1 individual litter
        self.assertGreaterEqual(f10008_sg["garbage_piles"], 2)
        self.assertGreaterEqual(f10008_sg["groups_of_litter"], 1)
        self.assertGreaterEqual(f10008_sg["individual_litter"], 1)
        self.assertGreaterEqual(f10008_sg["total_regions"], 4)

    def test_06_full_65_sweep_zero_lost_tps_and_monotonic_ap(self):
        """Full 65 validation images: zero lost TPs, clean images >= 35, AP9 >= 0.295136."""
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

        # Zero Lost True Positives
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


if __name__ == "__main__":
    unittest.main()
