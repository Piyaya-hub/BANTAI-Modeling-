"""Unit tests for Stage 5 Unmatched-Query Background Suppression & False-Positive Elimination (D-098)."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    apply_consensus_tiebreak,
    PLASTIC_SUBTYPE_IDS,
)
from scripts.taco_stage5_visual_review import discover_d098_run, D098_CHAMPION_STATE
from scripts.taco_stage5_material_hierarchy import evaluate_case, MAPS, NAMES, iou

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D098_RUN = ROOT / "runs/training/stage5-d098-bg-suppression-20261106-a"


class TestStage5D098BgSuppression(unittest.TestCase):

    def test_background_hard_negative_loss_logic(self):
        # Target score <= 0.06 -> zero penalty
        prob_9 = torch.zeros((1, 900, 9), dtype=torch.float32)
        prob_9[0, 10, 0] = 0.05  # background query cigarette score 0.05 <= 0.06
        loss_zero = torch.relu(prob_9[0, 10, 0] - 0.06).pow(2)
        self.assertAlmostEqual(float(loss_zero), 0.0, places=6)

        # Target score > 0.06 -> quadratic suppression penalty
        prob_9[0, 10, 0] = 0.25
        loss_hn = torch.relu(prob_9[0, 10, 0] - 0.06).pow(2)
        expected = (0.25 - 0.06) ** 2
        self.assertAlmostEqual(float(loss_hn), expected, places=6)

    def test_positive_match_floor_loss_logic(self):
        # Target score >= 0.35 -> zero penalty
        prob_9 = torch.zeros((1, 900, 9), dtype=torch.float32)
        prob_9[0, 20, 8] = 0.40  # paper target query at 0.40 >= 0.35
        loss_floor_zero = torch.relu(0.35 - prob_9[0, 20, 8]).pow(2)
        self.assertAlmostEqual(float(loss_floor_zero), 0.0, places=6)

        # Target score < 0.35 -> positive floor penalty
        prob_9[0, 20, 8] = 0.20
        loss_floor_pos = torch.relu(0.35 - prob_9[0, 20, 8]).pow(2)
        expected = (0.35 - 0.20) ** 2
        self.assertAlmostEqual(float(loss_floor_pos), expected, places=6)

    def test_image165_beach_false_positive_clutter_elimination(self):
        val_data = strict_load(VAL_JSON)
        img165_meta = next(img for img in val_data["images"] if img["id"] == 165)
        gt_ann = next(a for a in val_data["annotations"] if a["image_id"] == 165)

        d098_champ = strict_load(D098_RUN / "soup_d098_d093champ_5050.json")
        d098_vecs = np.load(D098_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d098_vecs["image_ids"])}

        img165_rows = [r for r in d098_champ["raw_queries"] if r["image_id"] == 165]
        w, h = img165_meta["width"], img165_meta["height"]
        vecs = d098_vecs["scores"][img_map[165]]

        adj, changed, suppressed, diag = apply_scene_consensus(
            img165_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            return_diagnostics=True,
        )

        passing_10 = [r for r in adj if r["score"] >= 0.10]

        # Verify exactly 1 passing detection remains at >= 0.10
        self.assertEqual(len(passing_10), 1)
        det = passing_10[0]

        # Verify it is the GT cigarette butt
        self.assertEqual(det["category_id"], 0)
        self.assertGreaterEqual(det["score"], 0.25)
        self.assertGreaterEqual(iou(det["bbox"], gt_ann["bbox"]), 0.50)

        # Verify all beach false positives (seashells, crab, pebbles, sand indentations) are eliminated
        plastic_boxes = [r for r in passing_10 if r["category_id"] in PLASTIC_SUBTYPE_IDS]
        self.assertEqual(len(plastic_boxes), 0)
        paper_boxes = [r for r in passing_10 if r["category_id"] == 8]
        self.assertEqual(len(paper_boxes), 0)

        # Rule 5 must have triggered
        self.assertGreater(diag["rule5_count"], 0)

    def test_image1_paper_preservation_zero_plastic_clutter(self):
        val_data = strict_load(VAL_JSON)
        img1_meta = next(img for img in val_data["images"] if img["id"] == 1)

        d098_champ = strict_load(D098_RUN / "soup_d098_d093champ_5050.json")
        d098_vecs = np.load(D098_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d098_vecs["image_ids"])}

        img1_rows = [r for r in d098_champ["raw_queries"] if r["image_id"] == 1]
        w, h = img1_meta["width"], img1_meta["height"]
        vecs = d098_vecs["scores"][img_map[1]]

        adj, changed, suppressed, diag = apply_scene_consensus(
            img1_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            return_diagnostics=True,
        )

        passing_10 = [r for r in adj if r["score"] >= 0.10]

        # Passing 0.10 boxes must be ONLY category_id == 8 (paper)
        self.assertEqual(len(passing_10), 3)
        self.assertTrue(all(r["category_id"] == 8 for r in passing_10))

        # Zero plastic or hazard boxes
        self.assertEqual(len([r for r in passing_10 if r["category_id"] != 8]), 0)

    def test_image777_cautious_preservation(self):
        val_data = strict_load(VAL_JSON)
        img777_meta = next(img for img in val_data["images"] if img["id"] == 777)
        gt408 = next(a for a in val_data["annotations"] if a["image_id"] == 777 and a["id"] == 408)

        d098_champ = strict_load(D098_RUN / "soup_d098_d093champ_5050.json")
        d098_vecs = np.load(D098_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d098_vecs["image_ids"])}

        img777_rows = [r for r in d098_champ["raw_queries"] if r["image_id"] == 777]
        w, h = img777_meta["width"], img777_meta["height"]
        vecs = d098_vecs["scores"][img_map[777]]

        adj, changed, suppressed, diag = apply_scene_consensus(
            img777_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
            return_diagnostics=True,
        )

        passing_10 = [r for r in adj if r["score"] >= 0.10]
        self.assertEqual(len(passing_10), 5)

        # Zero paper boxes
        paper_boxes = [r for r in passing_10 if r["category_id"] == 8]
        self.assertEqual(len(paper_boxes), 0)

        # 1 Plastic bottle / wrapper match
        bottle_matches = [r for r in passing_10 if iou(r["bbox"], gt408["bbox"]) >= 0.50]
        self.assertEqual(len(bottle_matches), 1)
        self.assertIn(bottle_matches[0]["category_id"], PLASTIC_SUBTYPE_IDS)

        # 4 Cigarette butts retained
        cigs = [r for r in passing_10 if r["category_id"] == 0]
        self.assertEqual(len(cigs), 4)

    def test_d098_zero_true_positives_lost_across_65_images(self):
        val_data = strict_load(VAL_JSON)
        d098_champ = strict_load(D098_RUN / "soup_d098_d093champ_5050.json")
        d098_vecs = np.load(D098_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d098_vecs["image_ids"])}
        img_dims = {img["id"]: (img["width"], img["height"]) for img in val_data["images"]}

        raw_by_img = {}
        for r in d098_champ["raw_queries"]:
            raw_by_img.setdefault(r["image_id"], []).append(r)

        lost_tps = []
        for img_id, rows in raw_by_img.items():
            w, h = img_dims[img_id]
            adj, _, _ = apply_scene_consensus(
                rows, d098_vecs["scores"][img_map[img_id]],
                image_width=w, image_height=h,
            )
            gts = [a for a in val_data["annotations"] if a["image_id"] == img_id]
            raw_passing = [r for r in rows if r["score"] >= 0.10]
            adj_passing = [r for r in adj if r["score"] >= 0.10]

            for gt in gts:
                raw_hit = any(r["category_id"] == gt["category_id"] and iou(r["bbox"], gt["bbox"]) >= 0.50 for r in raw_passing)
                adj_hit = any(r["category_id"] == gt["category_id"] and iou(r["bbox"], gt["bbox"]) >= 0.50 for r in adj_passing)
                if raw_hit and not adj_hit:
                    lost_tps.append({"image_id": img_id, "gt_id": gt["id"], "category_id": gt["category_id"]})

        self.assertEqual(lost_tps, [], f"Rule 5 must not lose any true positive: {lost_tps}")

    def test_d098_fold_a_and_fold_b_monotonic_improvement(self):
        val_data = strict_load(VAL_JSON)
        d098_champ = strict_load(D098_RUN / "soup_d098_d093champ_5050.json")
        d098_vecs = np.load(D098_RUN / "best-score-vectors.npz")
        img_map = {img_id: idx for idx, img_id in enumerate(d098_vecs["image_ids"])}
        img_dims = {img["id"]: (img["width"], img["height"]) for img in val_data["images"]}

        raw_rows_by_img = {}
        for r in d098_champ["raw_queries"]:
            raw_rows_by_img.setdefault(r["image_id"], []).append(r)

        sorted_img_ids = sorted(raw_rows_by_img.keys())
        fold_A_ids = [img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 0]
        fold_B_ids = [img_id for idx, img_id in enumerate(sorted_img_ids) if idx % 2 == 1]

        adj_rows_by_img = {}
        for img_id in sorted_img_ids:
            w, h = img_dims[img_id]
            adj, _, _ = apply_scene_consensus(
                raw_rows_by_img[img_id], d098_vecs["scores"][img_map[img_id]],
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

        self.assertGreaterEqual(ap9_adj_A, ap9_raw_A)
        self.assertGreaterEqual(ap5_adj_A, ap5_raw_A)

        # Fold B APs
        raw_B = [r for img_id in fold_B_ids for r in raw_rows_by_img[img_id]]
        adj_B = [r for img_id in fold_B_ids for r in adj_rows_by_img[img_id]]
        ap9_raw_B = evaluate_case(data_B, raw_B, "9-specific", "raw")["metrics"]["AP"]
        ap9_adj_B = evaluate_case(data_B, adj_B, "9-specific", "raw")["metrics"]["AP"]
        ap5_raw_B = evaluate_case(data_B, raw_B, "bantai-5-category", "raw")["metrics"]["AP"]
        ap5_adj_B = evaluate_case(data_B, adj_B, "bantai-5-category", "raw")["metrics"]["AP"]

        self.assertGreaterEqual(ap9_adj_B, ap9_raw_B)
        self.assertGreaterEqual(ap5_adj_B, ap5_raw_B)

    def test_d098_run_discovery_and_artifacts(self):
        discovered = discover_d098_run()
        self.assertIsNotNone(discovered)
        self.assertEqual(discovered.resolve(), D098_RUN.resolve())

        report = strict_load(D098_RUN / "report.json")
        self.assertEqual(report.get("authorization"), "D-098")
        self.assertEqual(report.get("status"), "complete")
        self.assertEqual(report.get("total_steps", report.get("training_updates")), 576)

        # Verify best inference bundle manifest
        soup_manifest = strict_load(D098_RUN / "checkpoints/best-inference/soup_manifest.json")
        self.assertEqual(soup_manifest.get("best_state"), "soup_d098_d093champ_5050")
        self.assertEqual(
            soup_manifest.get("parameter_sha256"),
            "77e3a8064f3c88b19b1aee1f288f5d216544a381da141ef8678edb4a7b5847ff",
        )

        # Verify score vectors shape
        vecs = np.load(D098_RUN / "best-score-vectors.npz")
        self.assertEqual(list(vecs["scores"].shape), [65, 900, 9])
        self.assertEqual(len(vecs["image_ids"]), 65)


if __name__ == "__main__":
    unittest.main()
