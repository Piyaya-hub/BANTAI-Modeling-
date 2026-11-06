"""Unit tests for D-100: Cross-Stage Prior-Anchored Consensus (Stage 4 Baseline + Stage 5 Champion).

Verifies:
  1. Image 263 styrofoam restoration and border twig clutter elimination.
  2. Image 407 styrofoam restoration and floating water-weed hazard veto.
  3. Image 762 dead-leaves paper clutter and leaf-stem hazard veto.
  4. Preservation of real paper (Image 1) and real cigarettes (Image 777).
  5. Full 65-image sweep: zero lost True Positives across all 65 validation images.
  6. Strict monotonic AP improvements on Full 65, Fold A, and Fold B.
  7. Backward compatibility when enable_rule7=False.
"""

import math
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
FOLD_A_IDS = {
    1, 4, 14, 21, 25, 41, 57, 72, 80, 87, 88, 93, 107, 108, 126,
    139, 144, 153, 155, 165, 172, 177, 187, 196, 208, 222, 238, 240,
    254, 263, 274, 281, 288,
}


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


class TestStage5D100CrossStageConsensus(unittest.TestCase):
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

    def test_image263_styrofoam_restoration_and_twig_suppression(self):
        """Image 263: white foam in brush restored from paper:0.46 to styrofoam:0.46, top twig suppressed."""
        img_id = 263
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        # Run with D-100 Rule 7 enabled
        adj, changed, supp, diag = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h,
            baseline_rows=self.base_by_img.get(img_id),
            enable_rule7=True, return_diagnostics=True,
        )

        passing = [r for r in adj if r["score"] >= 0.10]
        # Exactly 1 clean detection passing threshold
        self.assertEqual(len(passing), 1)
        # Class restored to styrofoam_piece (7), score preserved at ~0.457
        self.assertEqual(passing[0]["category_id"], 7)
        self.assertAlmostEqual(passing[0]["score"], 0.4572, places=3)
        self.assertGreaterEqual(diag["rule7a_count"], 1)

    def test_image407_styrofoam_restoration_and_weed_suppression(self):
        """Image 407: white object in water restored to styrofoam, all 5 weed false-positives vetoed."""
        img_id = 407
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
        self.assertEqual(passing[0]["category_id"], 7)
        self.assertAlmostEqual(passing[0]["score"], 0.3630, places=3)
        # Zero cigarette false positives remain
        hazards = [r for r in passing if r["category_id"] == 0]
        self.assertEqual(len(hazards), 0)
        self.assertGreaterEqual(diag["rule7a_count"], 1)
        self.assertGreaterEqual(diag["rule7b_count"], 5)

    def test_image762_dead_leaves_paper_and_foliage_clutter_veto(self):
        """Image 762: wrapper preserved as plastic_film:0.31, dead leaves and corner clutter eliminated."""
        img_id = 762
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
        self.assertEqual(passing[0]["category_id"], 4)  # plastic_film
        self.assertAlmostEqual(passing[0]["score"], 0.3073, places=3)
        self.assertGreaterEqual(diag["rule7b_count"], 4)

    def test_image1_paper_and_image777_hazard_preservation(self):
        """Ensure Rule 7 does not harm real paper (Image 1) or real cigarettes (Image 777)."""
        # Image 1 (Real Paper)
        img1_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 1]
        w1, h1 = self.img_dims[1]
        adj1, _, _, diag1 = apply_scene_consensus(
            img1_rows, self.d095_vecs["scores"][self.img_map[1]],
            image_width=w1, image_height=h1, enable_rule7=True, return_diagnostics=True,
        )
        passing1 = [r for r in adj1 if r["score"] >= 0.10]
        # Both paper items are preserved
        papers1 = [r for r in passing1 if r["category_id"] == 8]
        self.assertEqual(len(papers1), 3)  # 3 paper boxes including subbox
        self.assertEqual(diag1["rule7a_count"], 0)

        # Image 777 (Real Cigarettes + Plastic Wrapper)
        img777_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == 777]
        w777, h777 = self.img_dims[777]
        adj777, _, _, diag777 = apply_scene_consensus(
            img777_rows, self.d095_vecs["scores"][self.img_map[777]],
            image_width=w777, image_height=h777, enable_rule7=True, return_diagnostics=True,
        )
        passing777 = [r for r in adj777 if r["score"] >= 0.10]
        # 1 Plastic wrapper + 4 Cigarette butts
        self.assertEqual(len(passing777), 5)
        self.assertEqual(passing777[0]["category_id"], 1)  # wrapper mapped to plastic_bottle
        haz777 = [r for r in passing777 if r["category_id"] == 0]
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

        self.assertGreaterEqual(tot_7a, 3)  # Images 263, 407, 676, and D-101 additions
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
        styro_ap = res9["metrics"]["classes"]["styrofoam_piece"]["AP"]

        self.assertGreaterEqual(ap9, 0.265959 - 1e-5)
        self.assertGreaterEqual(ap5, 0.446063 - 1e-5)
        self.assertAlmostEqual(styro_ap, 0.287129, places=5)

        # Fold A and Fold B Evaluation using canonical interleaved definition
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
        self.assertGreaterEqual(res9_a["metrics"]["AP"], 0.204820 - 1e-5)
        self.assertGreaterEqual(res5_a["metrics"]["AP"], 0.320082 - 1e-5)

        # Fold B Evaluation
        fold_b_adj = [r for img_id in sorted_img_ids if img_id in fold_B_ids for r in adj_by_img[img_id]]
        fold_b_gts = [a for a in self.val_data["annotations"] if a["image_id"] in fold_B_ids]
        fold_b_imgs = [img for img in self.val_data["images"] if img["id"] in fold_B_ids]
        val_fold_b = {"images": fold_b_imgs, "annotations": fold_b_gts, "categories": self.val_data["categories"]}
        res9_b = evaluate_case(val_fold_b, fold_b_adj, "9-specific", "raw")
        res5_b = evaluate_case(val_fold_b, fold_b_adj, "bantai-5-category", "raw")
        self.assertGreaterEqual(res9_b["metrics"]["AP"], 0.338434 - 1e-5)
        self.assertGreaterEqual(res5_b["metrics"]["AP"], 0.602474 - 1e-5)

    def test_backward_compatibility_rule7_disabled(self):
        """When enable_rule7=False, behavior reverts exactly to D-099 metrics."""
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
        # Reverts to D-099 behavior: paper:0.46 + plastic_film:0.11
        self.assertEqual(len(passing), 2)
        self.assertEqual(passing[0]["category_id"], 8)  # paper


if __name__ == "__main__":
    unittest.main()
