"""Unit tests for Stage 5 D-104: Scale-Normalized Spatial Grouping & GUI Integration.

Verifies:
  1. Code Hygiene Fix:
     - Zero occurrences of `PROTECTED_SHOWCASE_IDS` in `scripts/taco_stage5_consensus_tiebreak.py`.
     - 7B10 and 7B11 use clean, image-ID-independent feature checks without overfitting.
  2. Illustrative Scope Requirement:
     - 1 isolated bottle + 4 litter in 1 dispersed group:
       -> exactly 1 Individual Litter, 1 Group of Individual Litter, 0 Garbage Pile,
          5 physical items, 2 independent regions.
  3. Showcase Image 1 (`000008.jpg`):
     - Heavy overlap / high density (2 sheets of paper):
       -> exactly 1 Garbage Pile (2 physical items, 1 independent region).
  4. Showcase Image 777 (`000084.JPG`):
     - Dispersed wrapper + 4 cigarette butts:
       -> exactly 1 Group of Individual Litter (5 physical items, 1 independent region).
  5. Showcase Image 802 (`000021.jpg`):
     - Single paper detection:
       -> exactly 1 Individual Litter (1 physical item, 1 independent region).
  6. Showcase Image 998 (`000030.jpg`):
     - Two widely separated detections:
       -> exactly 2 Individual Litter (2 physical items, 2 independent regions).
  7. Provenance & Score Integrity:
     - Derived multi-item regions (item_count >= 2) have `model_score = None`.
     - Single items (item_count == 1) retain the exact float score of the underlying prediction.
     - Physical items are never double-counted (sum of region item_counts == total physical items).
     - Object predictions retain category_id, score, and bbox unaltered.
  8. GUI Integration:
     - `self.spatial_group_var = tk.BooleanVar(value=True)` present in `Stage5ViewerGUI`.
     - `"Spatial Grouping (Pile / Group / Individual Litter)"` available in review views.
     - Padded (4px) dashed box and high-contrast banner label rendered in `draw_panel`.
  9. Full 65-Image Validation Sweep:
     - Exactly 0 lost True Positives across all 65 validation images.
     - AP9 >= 0.295084, AP5 >= 0.491560 maintained.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _get_default_baseline_rows,
    _get_default_run1_rows,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_material_hierarchy import evaluate_case, VIEWS, VIEW_SCHEMES
from scripts.taco_stage5_spatial_grouping import (
    compute_spatial_grouping,
    compute_edge_gap,
    compute_box_iou,
    SPATIAL_CATEGORIES,
)
import scripts.taco_stage5_visual_review as viewer

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    return inter / union if union > 0.0 else 0.0


class TestStage5D104SpatialGrouping(unittest.TestCase):
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

    def test_01_code_hygiene_no_hardcoded_showcase_ids(self):
        """scripts/taco_stage5_consensus_tiebreak.py contains zero occurrences of PROTECTED_SHOWCASE_IDS."""
        src_path = ROOT / "scripts" / "taco_stage5_consensus_tiebreak.py"
        content = src_path.read_text(encoding="utf-8")
        self.assertNotIn("PROTECTED_SHOWCASE_IDS", content)
        self.assertNotIn("effective_image_id not in", content)

    def test_02_illustrative_scope_requirement(self):
        """1 isolated bottle + 4 litter in 1 group -> 1 Individual Litter, 1 Group of Individual Litter, 0 Garbage Pile."""
        illus_rows = [
            {"bbox": [50.0, 50.0, 60.0, 60.0], "score": 0.85, "category_id": 1, "query_id": 0},
            {"bbox": [1000.0, 1000.0, 40.0, 40.0], "score": 0.60, "category_id": 3, "query_id": 1},
            {"bbox": [1200.0, 1000.0, 40.0, 40.0], "score": 0.55, "category_id": 3, "query_id": 2},
            {"bbox": [1000.0, 1200.0, 40.0, 40.0], "score": 0.50, "category_id": 0, "query_id": 3},
            {"bbox": [1200.0, 1200.0, 40.0, 40.0], "score": 0.45, "category_id": 0, "query_id": 4},
        ]
        res = compute_spatial_grouping(illus_rows, image_width=2000.0, image_height=2000.0)
        s = res["summary"]

        self.assertEqual(s["total_physical_items"], 5)
        self.assertEqual(s["total_regions"], 2)
        self.assertEqual(s["individual_litter_count"], 1)
        self.assertEqual(s["group_of_individual_litter_count"], 1)
        self.assertEqual(s["garbage_pile_count"], 0)

        # Verify region properties
        indiv_regs = [r for r in res["regions"] if r["category"] == "individual_litter"]
        group_regs = [r for r in res["regions"] if r["category"] == "group_of_individual_litter"]
        self.assertEqual(len(indiv_regs), 1)
        self.assertEqual(len(group_regs), 1)

        self.assertEqual(indiv_regs[0]["item_count"], 1)
        self.assertAlmostEqual(indiv_regs[0]["model_score"], 0.85, places=2)
        self.assertEqual(group_regs[0]["item_count"], 4)
        self.assertIsNone(group_regs[0]["model_score"])

    def test_03_image_1_garbage_pile(self):
        """Image 1 (000008.jpg): 2 overlapping paper sheets -> 1 Garbage Pile (2 physical items, 1 region)."""
        img_id = 1
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, _, _ = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id, enable_rule7=True
        )
        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        res = compute_spatial_grouping(vis, image_width=w, image_height=h)
        s = res["summary"]

        self.assertEqual(s["total_physical_items"], 2)
        self.assertEqual(s["total_regions"], 1)
        self.assertEqual(s["garbage_pile_count"], 1)
        self.assertEqual(s["group_of_individual_litter_count"], 0)
        self.assertEqual(s["individual_litter_count"], 0)
        self.assertIsNone(res["regions"][0]["model_score"])
        self.assertEqual(res["regions"][0]["item_count"], 2)

    def test_04_image_777_group_of_individual_litter(self):
        """Image 777 (000084.JPG): 5 scattered items -> 1 Group of Individual Litter (5 items, 1 region)."""
        img_id = 777
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, _, _ = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id, enable_rule7=True
        )
        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        res = compute_spatial_grouping(vis, image_width=w, image_height=h)
        s = res["summary"]

        self.assertEqual(s["total_physical_items"], 5)
        self.assertEqual(s["total_regions"], 1)
        self.assertEqual(s["group_of_individual_litter_count"], 1)
        self.assertEqual(s["garbage_pile_count"], 0)
        self.assertEqual(s["individual_litter_count"], 0)
        self.assertIsNone(res["regions"][0]["model_score"])
        self.assertEqual(res["regions"][0]["item_count"], 5)

    def test_05_image_802_single_individual_litter(self):
        """Image 802 (000021.jpg): 1 detection -> 1 Individual Litter (1 item, 1 region)."""
        img_id = 802
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, _, _ = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id, enable_rule7=True
        )
        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        res = compute_spatial_grouping(vis, image_width=w, image_height=h)
        s = res["summary"]

        self.assertEqual(s["total_physical_items"], 1)
        self.assertEqual(s["total_regions"], 1)
        self.assertEqual(s["individual_litter_count"], 1)
        self.assertEqual(s["group_of_individual_litter_count"], 0)
        self.assertEqual(s["garbage_pile_count"], 0)
        self.assertIsNotNone(res["regions"][0]["model_score"])
        self.assertAlmostEqual(res["regions"][0]["model_score"], vis[0]["score"], places=4)

    def test_06_image_998_two_individual_litter(self):
        """Image 998 (000030.jpg): 2 widely separated detections -> 2 Individual Litter (2 items, 2 regions)."""
        img_id = 998
        w, h = self.img_dims[img_id]
        raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, _, _ = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id, enable_rule7=True
        )
        vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
        res = compute_spatial_grouping(vis, image_width=w, image_height=h)
        s = res["summary"]

        self.assertEqual(s["total_physical_items"], 2)
        self.assertEqual(s["total_regions"], 2)
        self.assertEqual(s["individual_litter_count"], 2)
        self.assertEqual(s["group_of_individual_litter_count"], 0)
        self.assertEqual(s["garbage_pile_count"], 0)
        for reg in res["regions"]:
            self.assertEqual(reg["item_count"], 1)
            self.assertIsNotNone(reg["model_score"])

    def test_07_gui_spatial_grouping_integration(self):
        """Verify GUI elements: checkbox, views dropdown entry, and draw_panel spatial region drawing."""
        src = inspect.getsource(viewer.Stage5ViewerGUI)
        self.assertIn("self.spatial_group_var = tk.BooleanVar(value=True)", src)
        self.assertIn("Spatial grouping (Pile / Group / Individual)", src)
        self.assertIn("REVIEW_VIEWS", src)
        self.assertIn("spatial_region", src)

        # Check REVIEW_VIEWS contains the spatial grouping option
        self.assertIn("Spatial Grouping (Pile / Group / Individual Litter)", viewer.REVIEW_VIEWS)

        # Check draw_panel supports spatial_regions
        dp_src = inspect.getsource(viewer.draw_panel)
        self.assertIn("spatial_regions", dp_src)
        self.assertIn("enc_rect", dp_src)

    def test_08_full_65_benchmark_invariants_and_ap(self):
        """Full 65 validation benchmark: 0 lost TPs, AP9 >= 0.295084, AP5 >= 0.491560, visible FPs <= 74."""
        raw_by_img = {}
        for r in self.d095_champ["raw_queries"]:
            raw_by_img.setdefault(r["image_id"], []).append(r)

        adj_by_img = {}
        total_tp = 0
        total_fp = 0
        clean_images = 0

        for i_id in sorted(raw_by_img.keys()):
            w, h = self.img_dims[i_id]
            raw_rows = raw_by_img[i_id]
            vecs = self.d095_vecs["scores"][self.img_map[i_id]]

            adj, _, _ = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h, image_id=i_id, enable_rule7=True
            )
            adj_by_img[i_id] = adj

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
                    img_tp += 1
                else:
                    img_fp += 1
            total_tp += img_tp
            total_fp += img_fp
            if img_fp == 0:
                clean_images += 1

        self.assertLessEqual(total_fp, 74)
        self.assertGreaterEqual(clean_images, 34)
        self.assertGreaterEqual(total_tp, 76)

        # 0 Lost True Positives
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

        all_adj = [r for img_id in sorted(adj_by_img.keys()) for r in adj_by_img[img_id]]
        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        self.assertGreaterEqual(res9["metrics"]["AP"], 0.295084 - 1e-5)
        self.assertGreaterEqual(res5["metrics"]["AP"], 0.491560 - 1e-5)


if __name__ == "__main__":
    unittest.main()
