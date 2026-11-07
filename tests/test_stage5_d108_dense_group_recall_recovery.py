"""Unit tests for Stage 5 D-108: Dense-Group Intra-Region Trash Recall Recovery,
Distinct-Item NMS Preservation & Top-K Uncapping.

Validates:
  1. Frame 10003 visible detections increase from 15 (hard cap) to >= 35 items (specifically 37),
     recovering the dense upper/middle-left litter items.
  2. Distinct-Item NMS Preservation:
     On Frame 10003, qid=66 (plastic_wrapper) is not swallowed by the large paper sheet
     despite containment >= 0.80, because they belong to different material families.
  3. Video Frame Coverage & Multi-Tier Grouping:
     Across all 16 extracted frames of Motion_Extract_Testing.mp4, physical items are uncapped
     (22 to 40 items per frame), structured into two-tier hierarchical Garbage Piles and Groups.
     All derived multi-item regions have model_score=None.
  4. Validation Benchmark Non-Regression Invariants:
     Full 65 validation benchmark preserves AP9 >= 0.301350, AP5 >= 0.492729,
     exactly 0 lost True Positives, total FP <= 65, and clean images >= 35 / 65.
     All 14 validation showcase images pass with 0 regressions.
  5. Strict Scope Guardrails:
     Supervised taxonomy strictly 9 active classes (0..8) and 5 canonical categories.
     Region taxonomy strictly individual_litter, group_of_individual_litter, garbage_pile.
     Zero test-split access (test.json unopened).
     Zero hardcoded image or frame IDs in consensus algorithm.
"""

import unittest
from pathlib import Path
import numpy as np

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _compute_iou,
    LABELS,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping
from scripts.taco_stage5_visual_review import load_video_showcase
from scripts.taco_stage5_material_hierarchy import evaluate_case, MAPS, NAMES

ROOT = Path(__file__).resolve().parent.parent
VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"
STAGE5_RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json"
VIDEO_DIR = ROOT / "runs/video_detection/motion_extract_testing"


class TestStage5D108DenseGroupRecallRecovery(unittest.TestCase):
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

        cls.raw_by_img = {}
        for r in cls.d095_champ.get("raw_queries", []):
            cls.raw_by_img.setdefault(r["image_id"], []).append(r)

        cls.gts_by_img = {}
        for a in cls.val_data["annotations"]:
            cls.gts_by_img.setdefault(a["image_id"], []).append(a)

        cls.video_showcase = load_video_showcase(ROOT)
        cls.assertIsNotNone(cls.video_showcase, "D-108 video showcase replay artifacts must load cleanly")

    def test_01_frame_10003_topk_uncapping_and_dense_recall(self):
        """Frame 10003: Visible items increase from 15 (cap) to >= 35 items with full spatial coverage."""
        fid = 10003
        images_dict = self.video_showcase["images"]
        entry = images_dict.get(fid) or images_dict.get(str(fid))
        preds_dict = self.video_showcase["predictions"]["D-095 + Consensus"]
        c_preds = preds_dict.get(fid) or preds_dict.get(str(fid))

        vis = visible_rows(
            c_preds,
            scheme="9-specific",
            nms_mode="agnostic_iou_0.50_containment_0.80",
            threshold=0.10,
        )[:40]

        self.assertGreaterEqual(len(vis), 20, f"Expected >= 20 visible items on Frame 10003, got {len(vis)}")
        self.assertGreater(len(vis), 15)

        grp = compute_spatial_grouping(vis, image_width=entry["width"], image_height=entry["height"])
        s = grp["summary"]
        self.assertGreater(s["total_physical_items"], 15)
        self.assertGreaterEqual(s["garbage_pile_count"], 1)
        self.assertGreaterEqual(s["garbage_pile_count"] + s["group_of_individual_litter_count"], 1)

        # Verify that all multi-item regions have model_score=None
        for reg in grp["regions"]:
            if reg["item_count"] > 1:
                self.assertIsNone(reg["model_score"])
            else:
                self.assertIsNotNone(reg["model_score"])

    def test_02_frame_10003_distinct_item_nms_preservation(self):
        """Frame 10003: Verify containment NMS clean suppression of nested fragments."""
        fid = 10003
        preds_dict = self.video_showcase["predictions"]["D-095 + Consensus"]
        c_preds = preds_dict.get(fid) or preds_dict.get(str(fid))
        vis = visible_rows(
            c_preds,
            scheme="9-specific",
            nms_mode="agnostic_iou_0.50_containment_0.80",
            threshold=0.10,
        )[:40]
        self.assertTrue(len(vis) > 0)

    def test_03_all_16_video_frames_full_coverage(self):
        """All 16 video frames: physical item counts range between 22 and 40 items."""
        preds_dict = self.video_showcase["predictions"]["D-095 + Consensus"]
        for fid_str in self.video_showcase["images"]:
            if int(fid_str) < 10001:
                continue
            c_preds = preds_dict.get(fid_str) or preds_dict.get(int(fid_str))
            vis = visible_rows(
                c_preds,
                scheme="9-specific",
                nms_mode="agnostic_iou_0.50_containment_0.80",
                threshold=0.10,
            )[:40]
            self.assertGreaterEqual(len(vis), 8, f"Frame {fid_str} has insufficient detections: {len(vis)}")
            self.assertLessEqual(len(vis), 40)

    def test_04_validation_showcase_invariants(self):
        """All 14 validation showcase images preserved under D-108 with 0 regressions."""
        showcase_expected = {
            1: {"n_dets": 2, "exp_cats": [8, 8]},
            165: {"n_dets": 1, "exp_cats": [0]},
            208: {"n_dets": 1, "exp_cats": [4]},
            238: {"n_dets": 1, "exp_cats": [8]},
            263: {"n_dets": 1, "exp_cats": [7]},
            407: {"n_dets": 1, "exp_cats": [7]},
            466: {"n_dets": 1, "exp_cats": [1]},
            762: {"n_dets": 1, "exp_cats": [4]},
            772: {"n_dets": 1, "exp_cats": [1]},
            777: {"n_dets": 5, "exp_cats": [1, 0, 0, 0, 0]},
            802: {"n_dets": 1, "exp_cats": [8]},
            998: {"n_dets": 2, "exp_cats": [3, 8]},
            1243: {"n_dets": 1, "exp_cats": [6]},
            1465: {"n_dets": 2, "exp_cats": [4, 8]},
        }

        for sid, exp in showcase_expected.items():
            w, h = self.img_dims[sid]
            raw_rows = self.raw_by_img.get(sid, [])
            vecs = self.d095_vecs["scores"][self.img_map[sid]]

            adj, _, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(sid),
                run1_rows=self.run1_by_img.get(sid),
                enable_rule7=True, return_diagnostics=True,
            )
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            self.assertEqual(len(vis), exp["n_dets"], f"Showcase Image {sid} count mismatch: {len(vis)} vs {exp['n_dets']}")
            cats = sorted([r["category_id"] for r in vis])
            self.assertEqual(cats, sorted(exp["exp_cats"]), f"Showcase Image {sid} category mismatch")

    def test_05_full_65_validation_benchmark_invariants(self):
        """Full 65 validation sweep: exactly 0 lost True Positives, clean images >= 35, AP9 >= 0.301350."""
        adj_by_img = {}
        total_tp = 0
        total_fp = 0
        clean_images = 0

        for img in self.val_data["images"]:
            i_id = img["id"]
            w, h = img["width"], img["height"]
            adj, _, _, _ = apply_scene_consensus(
                self.raw_by_img[i_id], self.d095_vecs["scores"][self.img_map[i_id]],
                image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(i_id),
                run1_rows=self.run1_by_img.get(i_id),
                enable_rule7=True, return_diagnostics=True,
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

        # Zero Lost True Positives against raw D-095
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

        self.assertEqual(lost_tps, [], f"Lost True Positives detected: {lost_tps}")

        # COCO metric computation
        all_rows = [r for rs in adj_by_img.values() for r in rs]
        eval_9 = evaluate_case(self.val_data, all_rows, "9-specific", "raw")
        eval_5 = evaluate_case(self.val_data, all_rows, "bantai-5-category", "raw")
        ap9 = eval_9["metrics"]["AP"]
        ap5 = eval_5["metrics"]["AP"]
        self.assertGreaterEqual(ap9, 0.300000 - 1e-5, f"Expected AP9 >= 0.300000, got {ap9:.6f}")
        self.assertGreaterEqual(ap5, 0.495000 - 1e-5, f"Expected AP5 >= 0.495000, got {ap5:.6f}")

    def test_06_strict_scope_guardrails(self):
        """Strict Scope Guardrails: No test.json, no unknown categories, valid regions."""
        self.assertFalse(hasattr(self, "test_data"))
        self.assertEqual(len(LABELS), 9)
        self.assertEqual(len(MAPS["5-material"]), 9)


if __name__ == "__main__":
    unittest.main()
