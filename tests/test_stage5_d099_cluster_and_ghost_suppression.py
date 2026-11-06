"""Unit tests for Stage 5 Cluster Consolidation and Marginal Ghost Suppression (D-099)."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    PLASTIC_SUBTYPE_IDS,
)
from scripts.taco_stage5_material_hierarchy import evaluate_case, MAPS, NAMES, iou
from scripts.taco_stage5_display_profiles import visible_rows

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"


class TestStage5D099ClusterAndGhostSuppression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.val_data = strict_load(VAL_JSON)
        cls.d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        cls.d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        cls.img_map = {img_id: idx for idx, img_id in enumerate(cls.d095_vecs["image_ids"])}

    def test_image772_cluster_consolidation(self):
        """Image 772: 5 fragmented Plastic sub-boxes collapse into 1 clean Plastic 0.21 box."""
        i_id = 772
        meta = next(img for img in self.val_data["images"] if img["id"] == i_id)
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == i_id]
        vecs = self.d095_vecs["scores"][self.img_map[i_id]]

        # Without Rule 6:
        adj_legacy, _, _, diag_legacy = apply_scene_consensus(
            raw_rows, vecs, image_width=meta["width"], image_height=meta["height"],
            enable_rule6=False, return_diagnostics=True
        )
        self.assertEqual(diag_legacy["rule6a_count"], 0)
        self.assertEqual(diag_legacy["rule6b_count"], 0)

        # With Rule 6:
        adj_r6, _, _, diag_r6 = apply_scene_consensus(
            raw_rows, vecs, image_width=meta["width"], image_height=meta["height"],
            enable_rule6=True, return_diagnostics=True
        )
        self.assertGreaterEqual(diag_r6["rule6a_count"], 5)
        self.assertEqual(diag_r6["rule6b_count"], 0)

        # Under default containment NMS, exactly 1 box remains: Plastic 0.21
        vis = visible_rows(adj_r6, scheme="5-material", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1)
        self.assertEqual(vis[0]["category_id"], 1)  # Plastic
        self.assertAlmostEqual(vis[0]["score"], 0.211, places=2)

    def test_image208_marginal_macro_ghost_suppression(self):
        """Image 208: border-touching macro ghost (qid=2, sc=0.136) is suppressed by Rule 6B."""
        i_id = 208
        meta = next(img for img in self.val_data["images"] if img["id"] == i_id)
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == i_id]
        vecs = self.d095_vecs["scores"][self.img_map[i_id]]

        adj_r6, _, _, diag_r6 = apply_scene_consensus(
            raw_rows, vecs, image_width=meta["width"], image_height=meta["height"],
            enable_rule6=True, return_diagnostics=True
        )
        self.assertEqual(diag_r6["rule6b_count"], 1)

        # Under containment NMS, exactly 1 box remains: Plastic 0.34 (the true styrofoam piece)
        vis = visible_rows(adj_r6, scheme="5-material", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        self.assertEqual(len(vis), 1)
        self.assertEqual(vis[0]["category_id"], 1)  # Plastic
        self.assertAlmostEqual(vis[0]["score"], 0.344, places=2)

    def test_all_65_images_monotonic_improvement(self):
        """Full 65-image sweep: 0 TPs lost in 9-specific, AP monotonically improves, styrofoam AP preserved."""
        all_adj = []
        tot_6a = 0
        tot_6b = 0
        for img in self.val_data["images"]:
            i_id = img["id"]
            raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == i_id]
            vecs = self.d095_vecs["scores"][self.img_map[i_id]]
            adj, _, _, diag = apply_scene_consensus(
                raw_rows, vecs, image_width=img["width"], image_height=img["height"],
                enable_rule6=True, enable_rule7=False, return_diagnostics=True
            )
            tot_6a += diag["rule6a_count"]
            tot_6b += diag["rule6b_count"]
            all_adj.extend(adj)

        self.assertEqual(tot_6a, 18)
        self.assertEqual(tot_6b, 3)

        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        # 9-specific AP improves over baseline 0.241505
        self.assertGreater(res9["metrics"]["AP"], 0.2424)
        # 5-category AP improves over baseline 0.418866
        self.assertGreater(res5["metrics"]["AP"], 0.4190)
        # styrofoam_piece AP strictly preserved at 0.148515
        self.assertAlmostEqual(res9["metrics"]["classes"]["styrofoam_piece"]["AP"], 0.148515, places=5)

    def test_showcase_images_cleanliness(self):
        """All 5 showcase images show clean single-object or expected GT detections."""
        expected_counts = {
            1: (2, "Paper"),
            165: (1, "Hazard"),
            208: (1, "Plastic"),
            772: (1, "Plastic"),
            777: (5, "Plastic+Butts"),
        }
        for i_id, (expected_n, label) in expected_counts.items():
            meta = next(img for img in self.val_data["images"] if img["id"] == i_id)
            raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == i_id]
            vecs = self.d095_vecs["scores"][self.img_map[i_id]]
            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=meta["width"], image_height=meta["height"],
                enable_rule6=True
            )
            vis = visible_rows(adj, scheme="5-material", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            self.assertEqual(len(vis), expected_n, f"Failed on Image {i_id} ({label}): expected {expected_n}, got {len(vis)}")


if __name__ == "__main__":
    unittest.main()
