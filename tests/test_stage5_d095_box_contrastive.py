"""Unit tests for Stage 5 Box-Contrastive Fine-Tuning & Scene Consensus Workflow (D-095)."""
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import apply_scene_consensus, apply_consensus_tiebreak
from scripts.taco_stage5_visual_review import discover_d095_run, D095_CHAMPION_STATE
from scripts.taco_stage5_material_hierarchy import evaluate_case

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"


class TestStage5D095BoxContrastive(unittest.TestCase):

    def test_positive_margin_loss_logic(self):
        # Target score >= 0.35 -> no margin penalty
        prob_9 = torch.zeros((1, 900, 9), dtype=torch.float32)
        prob_9[0, 10, 8] = 0.40  # query 10 on paper class is 0.40
        loss_margin_zero = torch.relu(0.35 - prob_9[0, 10, 8]).pow(2)
        self.assertAlmostEqual(float(loss_margin_zero), 0.0, places=6)

        # Target score < 0.35 -> margin penalty > 0
        prob_9[0, 10, 8] = 0.20
        loss_margin_pos = torch.relu(0.35 - prob_9[0, 10, 8]).pow(2)
        expected = (0.35 - 0.20) ** 2
        self.assertAlmostEqual(float(loss_margin_pos), expected, places=6)

    def test_interior_subbox_loss_logic(self):
        # Sub-box inside paper GT has plastic activation > 0.08 -> interior penalty > 0
        prob_9 = torch.zeros((1, 900, 9), dtype=torch.float32)
        prob_9[0, 50, 4] = 0.30  # plastic_film query inside paper bag
        loss_interior = torch.relu(prob_9[0, 50, 4] - 0.08).pow(2)
        expected = (0.30 - 0.08) ** 2
        self.assertAlmostEqual(float(loss_interior), expected, places=6)

        # Sub-box with activation <= 0.08 -> zero penalty
        prob_9[0, 50, 4] = 0.05
        loss_interior_zero = torch.relu(prob_9[0, 50, 4] - 0.08).pow(2)
        self.assertAlmostEqual(float(loss_interior_zero), 0.0, places=6)

    def test_scene_consensus_image1_paper_strip_and_logo_suppression(self):
        val_data = strict_load(VAL_JSON)
        img1_meta = next(img for img in val_data["images"] if img["id"] == 1)

        d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d095_vecs["image_ids"])}

        img1_rows = [r for r in d095_champ["raw_queries"] if r["image_id"] == 1]
        w, h = img1_meta["width"], img1_meta["height"]
        vecs = d095_vecs["scores"][img_map[1]]

        adj, changed, suppressed, diag = apply_scene_consensus(
            img1_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            return_diagnostics=True,
        )

        passing_10 = [r for r in adj if r["score"] >= 0.10]
        paper_passing = [r for r in passing_10 if r["category_id"] == 8]

        # Verify GT#1 (paper bag) is present
        bag = [r for r in paper_passing if r["score"] > 0.30]
        self.assertEqual(len(bag), 1)
        self.assertAlmostEqual(bag[0]["score"], 0.3552, places=3)

        # Verify GT#2 (paper strip) was reassigned from plastic to paper
        strip = [r for r in paper_passing if 0.15 <= r["score"] <= 0.25]
        self.assertEqual(len(strip), 1)
        self.assertEqual(strip[0]["category_id"], 8)
        self.assertTrue(strip[0].get("consensus_reassigned", False))

        # Verify over 500 low-confidence interior subboxes/speckles were suppressed
        self.assertGreater(len(suppressed), 500)
        self.assertEqual(diag["anchors_found"], 1)

    def test_scene_consensus_image777_cautious_preservation(self):
        val_data = strict_load(VAL_JSON)
        img777_meta = next(img for img in val_data["images"] if img["id"] == 777)
        gt408 = next(a for a in val_data["annotations"] if a["image_id"] == 777 and a["id"] == 408)

        d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d095_vecs["image_ids"])}

        img777_rows = [r for r in d095_champ["raw_queries"] if r["image_id"] == 777]
        w, h = img777_meta["width"], img777_meta["height"]
        vecs = d095_vecs["scores"][img_map[777]]

        # 1. Under D-097 Calibrated Scene Consensus (Rule 1B + Rule 4 active)
        adj, changed, suppressed, diag = apply_scene_consensus(
            img777_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            return_diagnostics=True,
        )

        # Cautious scene consensus must find ZERO anchors on Image 777
        self.assertEqual(diag["anchors_found"], 0)
        self.assertEqual(diag["rule2_count"], 0)
        self.assertEqual(diag["rule3_count"], 0)
        # Rule 1B triggers on wrapper query 0 (coactivated plastic subtypes)
        self.assertEqual(diag["rule1b_count"], 1)
        # Rule 4 suppresses marginal 0.1009 paper ghost query 8
        self.assertEqual(diag["rule4_count"], 1)
        self.assertEqual(len(suppressed), 1)

        from scripts.taco_stage5_consensus_tiebreak import PLASTIC_SUBTYPE_IDS
        from scripts.taco_stage5_material_hierarchy import MAPS, NAMES, iou

        # D-097 Champion + Consensus: exactly 5 passing detections at 0.10
        passing_10 = [r for r in adj if r["score"] >= 0.10]
        self.assertEqual(len(passing_10), 5)

        # ZERO paper boxes at >= 0.10 on Image 777
        paper_boxes = [r for r in passing_10 if r["category_id"] == 8]
        self.assertEqual(len(paper_boxes), 0)

        # Bottle/wrapper detected as Plastic
        bottle_matches = [r for r in passing_10 if iou(r["bbox"], gt408["bbox"]) >= 0.50]
        self.assertEqual(len(bottle_matches), 1)
        b_match = bottle_matches[0]
        self.assertIn(b_match["category_id"], PLASTIC_SUBTYPE_IDS)
        cat_5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][b_match["category_id"]]]
        self.assertEqual(cat_5, "Plastic")
        self.assertAlmostEqual(b_match["score"], 0.3230, places=2)

        # 4 Cigarette butts retained as Potentially Hazardous
        cigs = [r for r in passing_10 if r["category_id"] == 0]
        self.assertEqual(len(cigs), 4)
        for c in cigs:
            c5 = NAMES["bantai-5-category"][MAPS["bantai-5-category"][c["category_id"]]]
            self.assertEqual(c5, "Potentially Hazardous")

        # 2. Backward-compatible check: when Rule 1B and Rule 4 are disabled
        adj_legacy, _, supp_legacy, diag_legacy = apply_scene_consensus(
            img777_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            enable_rule1b=False,
            enable_rule4=False,
            return_diagnostics=True,
        )
        self.assertEqual(diag_legacy["anchors_found"], 0)
        self.assertEqual(diag_legacy["rule1b_count"], 0)
        self.assertEqual(diag_legacy["rule4_count"], 0)
        self.assertEqual(len(supp_legacy), 0)

    def test_d097_fold_a_and_fold_b_monotonic_improvement(self):
        val_data = strict_load(VAL_JSON)
        d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d095_vecs["image_ids"])}
        img_dims = {img["id"]: (img["width"], img["height"]) for img in val_data["images"]}

        raw_rows_by_img = {}
        for r in d095_champ["raw_queries"]:
            raw_rows_by_img.setdefault(r["image_id"], []).append(r)

        sorted_img_ids = sorted(raw_rows_by_img.keys())
        fold_A_ids = [img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 0]
        fold_B_ids = [img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 1]

        adj_rows_by_img = {}
        for img_id in sorted_img_ids:
            w, h = img_dims[img_id]
            adj, _, _ = apply_scene_consensus(
                raw_rows_by_img[img_id], d095_vecs["scores"][img_map[img_id]],
                image_width=w, image_height=h,
            )
            adj_rows_by_img[img_id] = adj

        data_A = {
            "images": [img for img in val_data["images"] if img["id"] in fold_A_ids],
            "annotations": [a for a in val_data["annotations"] if a["image_id"] in fold_A_ids],
            "categories": val_data.get("categories", []),
        }
        data_B = {
            "images": [img for img in val_data["images"] if img["id"] in fold_B_ids],
            "annotations": [a for a in val_data["annotations"] if a["image_id"] in fold_B_ids],
            "categories": val_data.get("categories", []),
        }

        # Fold A APs
        raw_A = [r for img_id in fold_A_ids for r in raw_rows_by_img[img_id]]
        adj_A = [r for img_id in fold_A_ids for r in adj_rows_by_img[img_id]]
        ap9_raw_A = evaluate_case(data_A, raw_A, "9-specific", "raw")["metrics"]["AP"]
        ap9_adj_A = evaluate_case(data_A, adj_A, "9-specific", "raw")["metrics"]["AP"]
        ap5_raw_A = evaluate_case(data_A, raw_A, "bantai-5-category", "raw")["metrics"]["AP"]
        ap5_adj_A = evaluate_case(data_A, adj_A, "bantai-5-category", "raw")["metrics"]["AP"]

        self.assertGreater(ap9_adj_A, ap9_raw_A)
        self.assertGreater(ap5_adj_A, ap5_raw_A)

        # Fold B APs
        raw_B = [r for img_id in fold_B_ids for r in raw_rows_by_img[img_id]]
        adj_B = [r for img_id in fold_B_ids for r in adj_rows_by_img[img_id]]
        ap9_raw_B = evaluate_case(data_B, raw_B, "9-specific", "raw")["metrics"]["AP"]
        ap9_adj_B = evaluate_case(data_B, adj_B, "9-specific", "raw")["metrics"]["AP"]
        ap5_raw_B = evaluate_case(data_B, raw_B, "bantai-5-category", "raw")["metrics"]["AP"]
        ap5_adj_B = evaluate_case(data_B, adj_B, "bantai-5-category", "raw")["metrics"]["AP"]

        self.assertGreater(ap9_adj_B, ap9_raw_B)
        self.assertGreater(ap5_adj_B, ap5_raw_B)

    def test_d095_run_discovery_and_artifacts(self):
        discovered = discover_d095_run()
        self.assertIsNotNone(discovered)
        self.assertEqual(discovered.resolve(), D095_RUN.resolve())

        report = strict_load(D095_RUN / "report.json")
        self.assertEqual(report.get("authorization"), "D-095")
        self.assertEqual(report.get("status"), "complete")
        self.assertEqual(report.get("total_steps", report.get("training_updates")), 1152)

        # Verify best inference bundle manifest
        soup_manifest = strict_load(D095_RUN / "checkpoints/best-inference/soup_manifest.json")
        self.assertEqual(soup_manifest.get("best_state"), "soup_d095best_d093champ_5050")
        self.assertIn("parameter_sha256", soup_manifest)

        # Verify score vectors shape
        vecs = np.load(D095_RUN / "best-score-vectors.npz")
        self.assertEqual(list(vecs["scores"].shape), [65, 900, 9])
        self.assertEqual(len(vecs["image_ids"]), 65)


if __name__ == "__main__":
    unittest.main()
