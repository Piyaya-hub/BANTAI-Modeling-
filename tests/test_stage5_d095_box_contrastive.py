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

        adj, changed, suppressed = apply_scene_consensus(
            img1_rows, vecs, image_width=w, image_height=h,
            coactivation="3 plastic subtypes >= 0.20",
        )

        passing_10 = [r for r in adj if r["score"] >= 0.10]
        # Verify that all passing detections are Paper (cat=8)
        for r in passing_10:
            self.assertEqual(r["category_id"], 8)

        # Verify GT#1 (paper bag) is present
        bag = [r for r in passing_10 if r["score"] > 0.30]
        self.assertEqual(len(bag), 1)
        self.assertAlmostEqual(bag[0]["score"], 0.3552, places=3)

        # Verify GT#2 (paper strip) was reassigned from plastic to paper
        strip = [r for r in passing_10 if 0.15 <= r["score"] <= 0.25]
        self.assertEqual(len(strip), 1)
        self.assertEqual(strip[0]["category_id"], 8)
        self.assertTrue(strip[0].get("consensus_reassigned", False))

        # Verify zero false-positive cigarette speckles (cat=0) survived
        cig_passing = [r for r in passing_10 if r["category_id"] == 0]
        self.assertEqual(len(cig_passing), 0)

        # Verify zero false-positive plastic logo boxes (cat=4) survived
        film_passing = [r for r in passing_10 if r["category_id"] == 4]
        self.assertEqual(len(film_passing), 0)

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
