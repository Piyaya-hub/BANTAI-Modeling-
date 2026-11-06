"""Unit tests for D-102: Complete 65-Image False-Positive Elimination via Three-Model Consensus.

Three-Model Cross-Stage Consensus combines:
  1. Stage 4 Baseline (temporal anchor without paper overgeneralization)
  2. Stage 5 Run 1 (clean 9-class benchmark with real paper recall)
  3. Stage 5 D-095 Champion (high-recall unified detector)

Verifies:
  1. Image 238: GT=1 Paper. Secondary plastic clutter eliminated, exactly 1 box (Paper 0.393), 0 FP.
  2. Image 466: GT=1 Plastic. Dead foliage paper clutter eliminated, exactly 1 box (Plastic 0.282), 0 FP.
  3. Image 1465: GT=3. Duplicate paper fragment, shadow crack cigarette, and shadow rock paper eliminated.
     Exactly 2 visible detections (Plastic 0.303 + Paper 0.154), 0 FP.
  4. All 13 showcase images cleanly verified with zero regressions.
  5. Full 65-image sweep: exactly 0 lost True Positives across all 65 validation images.
  6. Strict monotonic AP gain across Full 65 (AP9 >= 0.284284, AP5 >= 0.472823), Fold A, and Fold B.
  7. Visible false positives reduced to <= 105, clean images increased to >= 31 / 65.
  8. Backward compatibility when enable_rule7=False.
"""

import unittest
from pathlib import Path

import numpy as np

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


class TestStage5D102ThreeModelConsensus(unittest.TestCase):
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

    def test_image238_paper_scene_secondary_plastic_eliminated(self):
        """Image 238: GT=1 Paper. Secondary plastic clutter eliminated, exactly 1 box (Paper 0.393), 0 FP."""
        img_id = 238
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            run1_rows=self.run1_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1)
        self.assertEqual(vis[0]["category_id"], 8)  # paper
        self.assertAlmostEqual(vis[0]["score"], 0.3928, places=3)

    def test_image466_dead_foliage_paper_clutter_eliminated(self):
        """Image 466: GT=1 Plastic. Dead foliage paper clutter eliminated, exactly 1 box (Plastic 0.282), 0 FP."""
        img_id = 466
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            run1_rows=self.run1_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1)
        self.assertEqual(vis[0]["category_id"], 1)  # plastic_bottle
        self.assertAlmostEqual(vis[0]["score"], 0.2815, places=3)

    def test_image1465_shadow_crack_and_rock_clutter_eliminated(self):
        """Image 1465: GT=3. Duplicate paper, shadow cigarette, and shadow rock paper eliminated. Exactly 2 boxes, 0 FP."""
        img_id = 1465
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            run1_rows=self.run1_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 2)
        cats = sorted(v["category_id"] for v in vis)
        self.assertEqual(cats, [4, 8])  # plastic_film (4), paper (8)

    def test_all_13_showcase_images_clean(self):
        """All 13 showcase images cleanly verified with zero false positives on single-object scenes."""
        expected_dets = {
            1: [(8, 0.355), (8, 0.194)],
            165: [(0, 0.282)],
            208: [(4, 0.344)],
            238: [(8, 0.393)],
            263: [(7, 0.457)],
            407: [(7, 0.363)],
            466: [(1, 0.282)],
            762: [(4, 0.307)],
            772: [(1, 0.211)],
            777: [(1, 0.323), (0, 0.133), (0, 0.128), (0, 0.116), (0, 0.115)],
            802: [(8, 0.281)],
            998: [(3, 0.442), (8, 0.315)],
            1465: [(4, 0.303), (8, 0.154)],
        }
        for sid, exp in expected_dets.items():
            w, h = self.img_dims[sid]
            raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == sid]
            vecs = self.d095_vecs["scores"][self.img_map[sid]]
            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(sid),
                run1_rows=self.run1_by_img.get(sid),
                enable_rule7=True,
            )
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            self.assertEqual(len(vis), len(exp), f"Image {sid} detection count mismatch")
            for det, (exp_cat, exp_sc) in zip(vis, exp):
                self.assertEqual(det["category_id"], exp_cat, f"Image {sid} category mismatch")
                self.assertAlmostEqual(det["score"], exp_sc, places=2, msg=f"Image {sid} score mismatch")

    def test_full_65_sweep_zero_lost_tps_and_monotonic_ap(self):
        """Full 65 images: exactly 0 lost TPs, monotonic AP gain across Full 65, Fold A, and Fold B."""
        raw_by_img = {}
        for r in self.d095_champ["raw_queries"]:
            raw_by_img.setdefault(r["image_id"], []).append(r)

        adj_by_img = {}
        total_tp = 0
        total_fp = 0
        clean_images = 0

        for img in self.val_data["images"]:
            i_id = img["id"]
            w, h = img["width"], img["height"]
            adj, _, _ = apply_scene_consensus(
                raw_by_img[i_id], self.d095_vecs["scores"][self.img_map[i_id]],
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

        # Visible FP reduction and clean image count
        self.assertLessEqual(total_fp, 105)
        self.assertGreaterEqual(clean_images, 31)

        # Verify 0 Lost True Positives
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

        self.assertGreaterEqual(ap9, 0.284284 - 1e-5)
        self.assertGreaterEqual(ap5, 0.472823 - 1e-5)
        self.assertGreaterEqual(paper_ap, 0.612674 - 1e-5)
        self.assertAlmostEqual(styro_ap, 0.287129, places=5)
        self.assertGreaterEqual(cig_ap, 0.124046 - 1e-5)

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
        self.assertGreaterEqual(res9_a["metrics"]["AP"], 0.230857 - 1e-5)
        self.assertGreaterEqual(res5_a["metrics"]["AP"], 0.360650 - 1e-5)

        # Fold B Evaluation
        fold_b_adj = [r for img_id in sorted_img_ids if img_id in fold_B_ids for r in adj_by_img[img_id]]
        fold_b_gts = [a for a in self.val_data["annotations"] if a["image_id"] in fold_B_ids]
        fold_b_imgs = [img for img in self.val_data["images"] if img["id"] in fold_B_ids]
        val_fold_b = {"images": fold_b_imgs, "annotations": fold_b_gts, "categories": self.val_data["categories"]}
        res9_b = evaluate_case(val_fold_b, fold_b_adj, "9-specific", "raw")
        res5_b = evaluate_case(val_fold_b, fold_b_adj, "bantai-5-category", "raw")
        self.assertGreaterEqual(res9_b["metrics"]["AP"], 0.339500 - 1e-5)
        self.assertGreaterEqual(res5_b["metrics"]["AP"], 0.605022 - 1e-5)

    def test_backward_compatibility_rule7_disabled(self):
        """When enable_rule7=False, behavior reverts cleanly."""
        img_id = 263
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            enable_rule7=False, return_diagnostics=True,
        )
        self.assertEqual(diag["rule7_count"], 0)
        self.assertEqual(diag["rule7a_count"], 0)
        self.assertEqual(diag["rule7b_count"], 0)
        passing = [r for r in adj if r["score"] >= 0.10]
        self.assertEqual(passing[0]["category_id"], 8)  # paper


if __name__ == "__main__":
    unittest.main()
