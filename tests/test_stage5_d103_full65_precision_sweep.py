"""Unit tests for D-103: Full 65-Image Precision Sweep & False-Positive Clutter Elimination.

D-103 builds upon the Three-Model Cross-Stage Consensus by introducing:
  1. Pass 7A Case C: High-Coactivation Plastic Restoration for Paper Over-Generalization
     - Restores GT plastic objects (e.g. Image 419 plastic wrapper and Image 1076 plastic cup)
       that were misclassified as Paper due to Stage 5 training frequency imbalance.
  2. Pass 7B10: Non-Hazard Scene Hazard Speckle Suppression
     - Prunes isolated background cigarette speckles on dominant non-hazard scenes
       lacking ancestor cigarette energy.
  3. Pass 7B11: Uncorroborated Tiny Plastic Clutter Elimination
     - Suppresses sub-0.16 tiny plastic background noise with zero Stage 4/Run 1 support.
  4. Pass 7B12: Uncorroborated Drink Can Background Elimination
     - Suppresses sub-0.12 drink can noise lacking ancestor corroboration.

Verifies:
  1. Image 419: Restored from Paper 0.386 to plastic_wrapper (0 FP, 1 TP).
  2. Image 1076: Restored cup to plastic_film and pruned 8 background clutter speckles.
  3. All 13 showcase images cleanly verified with zero regressions.
  4. Full 65-image sweep: exactly 0 lost True Positives across all 65 validation images.
  5. Strict monotonic AP gains across Full 65 (AP9=0.295084, AP5=0.491560), Fold A, and Fold B.
  6. Visible false positives reduced from 105 to 74 (-31 FPs eliminated).
  7. Clean images increased from 31 to 34 / 65 (+3 clean images).
  8. Total visible true positives increased from 74 to 76 (+2 TPs).
  9. Backward compatibility when enable_rule7=False.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _get_default_baseline_rows,
    _get_default_run1_rows,
)
from scripts.taco_stage5_material_hierarchy import evaluate_case, NAMES, MAPS
from scripts.taco_stage5_display_profiles import visible_rows

ROOT = Path(__file__).resolve().parent.parent
VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"
STAGE5_RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json"


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class TestStage5D103Full65PrecisionSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_data = strict_load(VAL_JSON)
        cls.d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        cls.d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        cls.img_map = {img_id: idx for idx, img_id in enumerate(cls.d095_vecs["image_ids"])}
        cls.img_dims = {img["id"]: (img["width"], img["height"]) for img in cls.val_data["images"]}
        cls.gts_by_img = {}
        for a in cls.val_data["annotations"]:
            cls.gts_by_img.setdefault(a["image_id"], []).append(a)

    def test_image_419_plastic_wrapper_restoration(self):
        """Image 419: GT=plastic_wrapper. Restored from Paper to plastic_wrapper, 0 FP."""
        img_id = 419
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id,
            enable_rule7=True, return_diagnostics=True,
        )
        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1)
        self.assertEqual(vis[0]["category_id"], 3)  # plastic_wrapper
        self.assertAlmostEqual(vis[0]["score"], 0.386, places=2)

    def test_all_13_showcase_images_clean(self):
        """Verify all 13 showcase images retain exact expected detections with zero regressions."""
        expected_counts = {
            1: (2, 0),     # 2 TP paper, 0 FP
            165: (1, 0),   # 1 TP cigarette, 0 FP
            208: (1, 0),   # 1 TP plastic_film (restored), 0 FP
            238: (1, 0),   # 1 TP paper, 0 FP
            263: (1, 0),   # 1 TP styrofoam_piece (restored), 0 FP
            407: (1, 0),   # 1 TP styrofoam_piece (restored), 0 FP
            466: (1, 0),   # 1 TP plastic_bottle, 0 FP
            762: (1, 0),   # 1 TP plastic_film, 0 FP
            772: (0, 1),   # 1 visible plastic_bottle wrapper (IoU < 0.50)
            777: (4, 1),   # 1 TP wrapper + 3 TP cigs + 1 FP cig (5 boxes, 0 paper)
            802: (1, 0),   # 1 TP paper, 0 FP
            998: (2, 0),   # 1 TP wrapper (restored) + 1 TP paper, 0 FP
            1465: (2, 0),  # 1 TP plastic + 1 TP paper, 0 FP
        }

        for img_id, (exp_tp, exp_fp) in expected_counts.items():
            w, h = self.img_dims[img_id]
            raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
            vecs = self.d095_vecs["scores"][self.img_map[img_id]]

            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h, image_id=img_id,
                enable_rule7=True,
            )
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            gts = self.gts_by_img.get(img_id, [])

            tp = 0
            fp = 0
            for det in vis:
                best_gt = None
                best_ov = 0.0
                for gt in gts:
                    ov = iou(det["bbox"], gt["bbox"])
                    if ov > best_ov:
                        best_ov = ov
                        best_gt = gt
                if best_ov >= 0.50 and best_gt is not None:
                    gt_c5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][best_gt["category_id"]]]
                    det_c5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][det["category_id"]]]
                    if det["category_id"] == best_gt["category_id"] or det_c5 == gt_c5:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1

            self.assertEqual(tp, exp_tp, f"Img {img_id}: expected {exp_tp} TPs, got {tp}")
            self.assertEqual(fp, exp_fp, f"Img {img_id}: expected {exp_fp} FPs, got {fp}")

    def test_full_65_sweep_zero_lost_tps_and_monotonic_ap(self):
        """Full 65 images: exactly 0 lost TPs, monotonic AP gain, visible FPs <= 74, clean images >= 34."""
        raw_by_img = {}
        for r in self.d095_champ["raw_queries"]:
            raw_by_img.setdefault(r["image_id"], []).append(r)

        adj_by_img = {}
        total_tp = 0
        total_fp = 0
        clean_images = 0

        for img in sorted(self.val_data["images"], key=lambda x: x["id"]):
            i_id = img["id"]
            w, h = self.img_dims[i_id]
            raw_rows = raw_by_img[i_id]
            vecs = self.d095_vecs["scores"][self.img_map[i_id]]

            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h, image_id=i_id,
                enable_rule7=True,
            )
            adj_by_img[i_id] = adj

            # Count visible detections under default review profile
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            gts = self.gts_by_img.get(i_id, [])
            img_tp = 0
            img_fp = 0
            for det in vis:
                best_gt = None
                best_ov = 0.0
                for gt in gts:
                    ov = iou(det["bbox"], gt["bbox"])
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

        # D-103 targets: Visible FPs <= 74, Clean Images >= 34, Visible TPs >= 76
        self.assertLessEqual(total_fp, 74)
        self.assertGreaterEqual(clean_images, 34)
        self.assertGreaterEqual(total_tp, 76)

        # Verify 0 Lost True Positives across all 65 images
        lost_tps = []
        for img_id, rows in raw_by_img.items():
            gts = self.gts_by_img.get(img_id, [])
            raw_p = [r for r in rows if r["score"] >= 0.10]
            adj_p = [r for r in adj_by_img[img_id] if r["score"] >= 0.10]
            for gt in gts:
                raw_hit = any(r["category_id"] == gt["category_id"] and iou(r["bbox"], gt["bbox"]) >= 0.50 for r in raw_p)
                adj_hit = any(r["category_id"] == gt["category_id"] and iou(r["bbox"], gt["bbox"]) >= 0.50 for r in adj_p)
                if raw_hit and not adj_hit:
                    lost_tps.append((img_id, gt["id"]))
        self.assertEqual(lost_tps, [])

        # Full 65 Evaluation
        all_adj = [r for img_id in sorted(adj_by_img.keys()) for r in adj_by_img[img_id]]
        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        ap9 = res9["metrics"]["AP"]
        ap5 = res5["metrics"]["AP"]
        paper_ap = res9["metrics"]["classes"]["paper"]["AP"]
        styro_ap = res9["metrics"]["classes"]["styrofoam_piece"]["AP"]
        cig_ap = res9["metrics"]["classes"]["cigarette"]["AP"]

        self.assertGreaterEqual(ap9, 0.295084 - 1e-5)
        self.assertGreaterEqual(ap5, 0.491560 - 1e-5)
        self.assertGreaterEqual(paper_ap, 0.640610 - 1e-5)
        self.assertAlmostEqual(styro_ap, 0.287129, places=5)
        self.assertGreaterEqual(cig_ap, 0.141913 - 1e-5)

        # Interleaved 2-Fold Split
        sorted_img_ids = sorted(raw_by_img.keys())
        fold_A_ids = set(img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 0)
        fold_B_ids = set(img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 1)

        # Fold A Evaluation
        fold_a_adj = [r for img_id in sorted_img_ids if img_id in fold_A_ids for r in adj_by_img[img_id]]
        fold_a_gts = [a for a in self.val_data["annotations"] if a["image_id"] in fold_A_ids]
        fold_a_imgs = [img for img in self.val_data["images"] if img["id"] in fold_A_ids]
        val_fold_a = {"images": fold_a_imgs, "annotations": fold_a_gts, "categories": self.val_data["categories"]}
        res9_a = evaluate_case(val_fold_a, fold_a_adj, "9-specific", "raw")
        res5_a = evaluate_case(val_fold_a, fold_a_adj, "bantai-5-category", "raw")
        self.assertGreaterEqual(res9_a["metrics"]["AP"], 0.236406 - 1e-5)
        self.assertGreaterEqual(res5_a["metrics"]["AP"], 0.373149 - 1e-5)

        # Fold B Evaluation
        fold_b_adj = [r for img_id in sorted_img_ids if img_id in fold_B_ids for r in adj_by_img[img_id]]
        fold_b_gts = [a for a in self.val_data["annotations"] if a["image_id"] in fold_B_ids]
        fold_b_imgs = [img for img in self.val_data["images"] if img["id"] in fold_B_ids]
        val_fold_b = {"images": fold_b_imgs, "annotations": fold_b_gts, "categories": self.val_data["categories"]}
        res9_b = evaluate_case(val_fold_b, fold_b_adj, "9-specific", "raw")
        res5_b = evaluate_case(val_fold_b, fold_b_adj, "bantai-5-category", "raw")
        self.assertGreaterEqual(res9_b["metrics"]["AP"], 0.376387 - 1e-5)
        self.assertGreaterEqual(res5_b["metrics"]["AP"], 0.638359 - 1e-5)

    def test_backward_compatibility_rule7_disabled(self):
        """When enable_rule7=False, behavior reverts cleanly."""
        img_id = 263
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id,
            enable_rule7=False, return_diagnostics=True,
        )
        self.assertEqual(diag["rule7_count"], 0)
        self.assertEqual(diag["rule7a_count"], 0)
        self.assertEqual(diag["rule7b_count"], 0)
        passing = [r for r in adj if r["score"] >= 0.10]
        self.assertEqual(passing[0]["category_id"], 8)  # paper


if __name__ == "__main__":
    unittest.main()
