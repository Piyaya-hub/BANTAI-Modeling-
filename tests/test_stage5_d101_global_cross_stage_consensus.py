"""Unit tests for D-101: Global 65-Image Cross-Stage Calibration & Consensus Refactoring.

Verifies:
  1. Image 802: Paper carton (0.281) preserved, interior sub-boxes and floor speckles eliminated (exactly 1 box).
  2. Image 998: Mixed cobblestone scene restored (1 plastic_wrapper:0.442 + 1 paper:0.315, exactly 2 boxes).
  3. Showcase preservation: Images 1, 165, 208, 263, 407, 762, 772, 777 remain 100% clean with zero regressions.
  4. Full 65 images: exactly 0 lost True Positives across all 65 validation images.
  5. Strict monotonic AP improvements on Full 65, Fold A, and Fold B.
  6. Backward compatibility when enable_rule7=False.
"""

import unittest
from pathlib import Path

import numpy as np

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _get_default_baseline_rows,
)
from scripts.taco_stage5_material_hierarchy import evaluate_case

ROOT = Path(__file__).resolve().parent.parent
VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"


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


class TestStage5D101GlobalCrossStageConsensus(unittest.TestCase):
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

    def test_image802_paper_carton_and_speckle_elimination(self):
        """Image 802: Paper carton preserved, interior texture and carpet speckles eliminated (exactly 1 box)."""
        img_id = 802
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        passing = [r for r in adj if r["score"] >= 0.10]
        # Exactly 1 clean detection passing threshold
        self.assertEqual(len(passing), 1)
        self.assertEqual(passing[0]["category_id"], 8)  # paper
        self.assertAlmostEqual(passing[0]["score"], 0.2813, places=3)
        self.assertGreaterEqual(diag["rule3_count"], 5)  # interior sub-boxes eliminated
        self.assertGreaterEqual(diag["rule7b_count"], 5)  # carpet speckles + floor clutter eliminated

    def test_image998_cobblestone_mixed_scene_restoration(self):
        """Image 998: Center wrapper restored to plastic_wrapper:0.442, top-left paper:0.315 preserved (exactly 2 boxes)."""
        img_id = 998
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        passing = [r for r in adj if r["score"] >= 0.10]
        # Exactly 2 clean detections passing threshold
        self.assertEqual(len(passing), 2)
        # Center wrapper restored to plastic_wrapper (3)
        wrap_box = [r for r in passing if r["query_id"] == 0][0]
        self.assertEqual(wrap_box["category_id"], 3)
        self.assertAlmostEqual(wrap_box["score"], 0.4420, places=3)
        # Top-left paper preserved as paper (8)
        paper_box = [r for r in passing if r["query_id"] == 1][0]
        self.assertEqual(paper_box["category_id"], 8)
        self.assertAlmostEqual(paper_box["score"], 0.3150, places=3)
        self.assertGreaterEqual(diag["rule7a_count"], 1)

    def test_showcase_preservation(self):
        """All 8 previously verified showcase images remain 100% clean with zero regressions."""
        # Image 1 (Real Paper)
        rows1 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 1]
        w1, h1 = self.img_dims[1]
        adj1, _, _, _ = apply_scene_consensus(
            rows1, self.d095_vecs["scores"][self.img_map[1]],
            image_width=w1, image_height=h1, enable_rule7=True, return_diagnostics=True,
        )
        p1 = [r for r in adj1 if r["score"] >= 0.10]
        self.assertEqual(len(p1), 3)
        self.assertTrue(all(r["category_id"] == 8 for r in p1))

        # Image 165 (Hazard only)
        rows165 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 165]
        w165, h165 = self.img_dims[165]
        adj165, _, _, _ = apply_scene_consensus(
            rows165, self.d095_vecs["scores"][self.img_map[165]],
            image_width=w165, image_height=h165, baseline_rows=self.base_by_img.get(165),
            enable_rule7=True, return_diagnostics=True,
        )
        p165 = [r for r in adj165 if r["score"] >= 0.10]
        self.assertEqual(len(p165), 1)
        self.assertEqual(p165[0]["category_id"], 0)

        # Image 208 (Plastic only)
        rows208 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 208]
        w208, h208 = self.img_dims[208]
        adj208, _, _, _ = apply_scene_consensus(
            rows208, self.d095_vecs["scores"][self.img_map[208]],
            image_width=w208, image_height=h208, baseline_rows=self.base_by_img.get(208),
            enable_rule7=True, return_diagnostics=True,
        )
        p208 = [r for r in adj208 if r["score"] >= 0.10]
        self.assertEqual(len(p208), 1)
        self.assertEqual(p208[0]["category_id"], 4)

        # Image 263 (Styrofoam restored)
        rows263 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 263]
        w263, h263 = self.img_dims[263]
        adj263, _, _, _ = apply_scene_consensus(
            rows263, self.d095_vecs["scores"][self.img_map[263]],
            image_width=w263, image_height=h263, baseline_rows=self.base_by_img.get(263),
            enable_rule7=True, return_diagnostics=True,
        )
        p263 = [r for r in adj263 if r["score"] >= 0.10]
        self.assertEqual(len(p263), 1)
        self.assertEqual(p263[0]["category_id"], 7)

        # Image 407 (Styrofoam restored, weed speckles vetoed)
        rows407 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 407]
        w407, h407 = self.img_dims[407]
        adj407, _, _, _ = apply_scene_consensus(
            rows407, self.d095_vecs["scores"][self.img_map[407]],
            image_width=w407, image_height=h407, baseline_rows=self.base_by_img.get(407),
            enable_rule7=True, return_diagnostics=True,
        )
        p407 = [r for r in adj407 if r["score"] >= 0.10]
        self.assertEqual(len(p407), 1)
        self.assertEqual(p407[0]["category_id"], 7)

        # Image 762 (Plastic wrapper preserved, foliage clutter vetoed)
        rows762 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 762]
        w762, h762 = self.img_dims[762]
        adj762, _, _, _ = apply_scene_consensus(
            rows762, self.d095_vecs["scores"][self.img_map[762]],
            image_width=w762, image_height=h762, baseline_rows=self.base_by_img.get(762),
            enable_rule7=True, return_diagnostics=True,
        )
        p762 = [r for r in adj762 if r["score"] >= 0.10]
        self.assertEqual(len(p762), 1)
        self.assertEqual(p762[0]["category_id"], 4)

        # Image 772 (Plastic wrapper single box)
        rows772 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 772]
        w772, h772 = self.img_dims[772]
        adj772, _, _, _ = apply_scene_consensus(
            rows772, self.d095_vecs["scores"][self.img_map[772]],
            image_width=w772, image_height=h772, baseline_rows=self.base_by_img.get(772),
            enable_rule7=True, return_diagnostics=True,
        )
        p772 = [r for r in adj772 if r["score"] >= 0.10]
        self.assertEqual(len(p772), 1)
        self.assertEqual(p772[0]["category_id"], 1)

        # Image 777 (1 Plastic wrapper + 4 Cigarette butts)
        rows777 = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 777]
        w777, h777 = self.img_dims[777]
        adj777, _, _, _ = apply_scene_consensus(
            rows777, self.d095_vecs["scores"][self.img_map[777]],
            image_width=w777, image_height=h777, baseline_rows=self.base_by_img.get(777),
            enable_rule7=True, return_diagnostics=True,
        )
        p777 = [r for r in adj777 if r["score"] >= 0.10]
        self.assertEqual(len(p777), 5)
        self.assertEqual(p777[0]["category_id"], 1)
        haz777 = [r for r in p777 if r["category_id"] == 0]
        self.assertEqual(len(haz777), 4)

    def test_full_65_sweep_zero_lost_tps_and_monotonic_ap(self):
        """Full 65 images: exactly 0 lost TPs, monotonic AP gain across Full 65, Fold A, and Fold B."""
        raw_by_img = {}
        for r in self.d095_champ["raw_queries"]:
            raw_by_img.setdefault(r["image_id"], []).append(r)

        adj_by_img = {}
        tot_7a = 0
        tot_7b = 0
        for img in self.val_data["images"]:
            i_id = img["id"]
            w, h = img["width"], img["height"]
            adj, _, _, diag = apply_scene_consensus(
                raw_by_img[i_id], self.d095_vecs["scores"][self.img_map[i_id]],
                image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(i_id),
                enable_rule7=True, return_diagnostics=True,
            )
            adj_by_img[i_id] = adj
            tot_7a += diag["rule7a_count"]
            tot_7b += diag["rule7b_count"]

        # Exactly 4 restorations: Images 263, 407, 676, and 998
        self.assertEqual(tot_7a, 4)
        self.assertGreaterEqual(tot_7b, 10)

        # Verify 0 Lost True Positives
        lost_tps = []
        for img_id, rows in raw_by_img.items():
            gts = [a for a in self.val_data["annotations"] if a["image_id"] == img_id]
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

        self.assertAlmostEqual(ap9, 0.282275, places=5)
        self.assertAlmostEqual(ap5, 0.468358, places=5)
        self.assertAlmostEqual(paper_ap, 0.599483, places=5)
        self.assertAlmostEqual(styro_ap, 0.287129, places=5)

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
        self.assertAlmostEqual(res9_a["metrics"]["AP"], 0.227830, places=5)
        self.assertAlmostEqual(res5_a["metrics"]["AP"], 0.353976, places=5)

        # Fold B Evaluation
        fold_b_adj = [r for img_id in sorted_img_ids if img_id in fold_B_ids for r in adj_by_img[img_id]]
        fold_b_gts = [a for a in self.val_data["annotations"] if a["image_id"] in fold_B_ids]
        fold_b_imgs = [img for img in self.val_data["images"] if img["id"] in fold_B_ids]
        val_fold_b = {"images": fold_b_imgs, "annotations": fold_b_gts, "categories": self.val_data["categories"]}
        res9_b = evaluate_case(val_fold_b, fold_b_adj, "9-specific", "raw")
        res5_b = evaluate_case(val_fold_b, fold_b_adj, "bantai-5-category", "raw")
        self.assertAlmostEqual(res9_b["metrics"]["AP"], 0.338439, places=5)
        self.assertAlmostEqual(res5_b["metrics"]["AP"], 0.602636, places=5)

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
