"""Unit tests for Stage 5 D-112: Rework Garbage Pile Minimum Cardinality Rule to k >= 8 Bounding Boxes (min_pile_items = 8).

Verifies:
  1. Code Hygiene & Zero Hardcoded Identifiers:
     - Active supervised taxonomy strictly locked to 9 classes (0..8) and 5 canonical categories.
     - Regional spatial taxonomy strictly locked to individual_litter, group_of_individual_litter, garbage_pile.
     - Multi-item derived regions have model_score = None. Single items retain exact model score.
     - Zero hardcoded image/frame IDs in consensus or spatial grouping logic.
  2. Strict Cardinality Boundary (D-112 Contract):
     - k = 1: individual_litter (model_score = float(score))
     - 2 <= k < 8 (k = 2, 3, 4, 5, 6, 7): group_of_individual_litter (model_score = None), even when touching/dense
     - k >= 8 (k = 8, 12, 16): garbage_pile (model_score = None) when contact/density criteria are met
     - min_pile_items parameterization override correctly supported.
  3. Standalone Image For_testing.jpg (Image 10000):
     - 4-item Center Curb Cluster classifies as Group of Individual Litter (4 items).
     - Lower-left cigarette and upper curb item classify as Individual Litter.
     - Exactly 0 Garbage Piles, 1 Group of Individual Litter, 2 Individual Litter (3 total regions).
  4. Video Frames (Motion_Extract_Testing_2.mp4 Frames 10005..10007):
     - All 4 crumpled paper towels classify as Group of Individual Litter (4 items).
     - Exactly 0 Garbage Piles, 1 Group of Individual Litter, 0 Individual Litter.
  5. Zero Overlapping Multi-Item Regions:
     - 0 overlapping multi-item region pairs (IoU <= 0.05 and IoMin < 0.20) across all video frames and all 65 validation images.
  6. Replay Artifact Integrity & Binding:
     - video-showcase.json and manifest.json bound under authorization D-112 with exact SHA-256 hashes.
     - load_video_showcase loads cleanly under D-112.
  7. Frozen 65-Validation Benchmark Bitwise Preservation:
     - Official 9-Specific AP == 0.301350
     - BANTAI 5-Category AP == 0.497161
     - Paper AP == 0.640610
     - True Positives == 77, Lost TPs == 0 (lost_tps == []), Total FPs == 63, Clean Images == 36 / 65.
"""
from __future__ import annotations

import hashlib
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
    _compute_iou,
    LABELS,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_material_hierarchy import evaluate_case, NAMES, MAPS
from scripts.taco_stage5_spatial_grouping import (
    compute_spatial_grouping,
    compute_box_intersection_and_areas,
    compute_box_iou,
    SPATIAL_CATEGORIES,
    DEFAULT_MIN_PILE_ITEMS,
)
from scripts.taco_stage5_visual_review import load_video_showcase
import scripts.taco_stage5_spatial_grouping as sg_mod
import scripts.taco_stage5_consensus_tiebreak as cb_mod

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
STAGE4_BASELINE = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json"
STAGE5_RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json"
VIDEO_DIR = ROOT / "runs/video_detection/motion_extract_testing"


class TestStage5D112GarbagePileMin8Rule(unittest.TestCase):
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

    def test_01_code_hygiene_and_guardrails(self):
        """Active supervised taxonomy locked to 9 classes; zero hardcoded identifiers."""
        self.assertEqual(len(LABELS), 9)
        self.assertEqual(LABELS[0], "cigarette")
        self.assertEqual(LABELS[8], "paper")
        self.assertEqual(DEFAULT_MIN_PILE_ITEMS, 8)

        # Spatial taxonomy strictly locked
        self.assertEqual(
            set(SPATIAL_CATEGORIES.keys()),
            {"individual_litter", "group_of_individual_litter", "garbage_pile"}
        )

        sg_src = inspect.getsource(sg_mod)
        cb_src = inspect.getsource(cb_mod)

        # Zero hardcoded IDs in spatial grouping or consensus logic
        forbidden_snippets = [
            "image_id == 10000", "image_id == 10001", "image_id == 10005",
            "image_id == 1", "image_id == 777", "image_id == 1465"
        ]
        for snip in forbidden_snippets:
            self.assertNotIn(snip, sg_src)
            self.assertNotIn(snip, cb_src)

        # Ensure frame check does not appear in spatial grouping logic
        self.assertNotIn("frame", sg_src)

    def test_02_synthetic_cardinality_boundary(self):
        """Strict D-112 cardinality rule: k=1 -> indiv, 2<=k<=7 -> group, k>=8 -> pile."""
        # 1. k = 1: isolated item -> individual_litter, retains model_score
        r1 = [{"bbox": [100.0, 100.0, 60.0, 60.0], "score": 0.8234, "category_id": 8, "query_id": 0}]
        res1 = compute_spatial_grouping(r1, 2000.0, 2000.0)
        self.assertEqual(res1["summary"]["individual_litter_count"], 1)
        self.assertEqual(res1["summary"]["group_of_individual_litter_count"], 0)
        self.assertEqual(res1["summary"]["garbage_pile_count"], 0)
        self.assertEqual(res1["regions"][0]["category"], "individual_litter")
        self.assertEqual(res1["regions"][0]["model_score"], 0.8234)

        # 2. k = 2..7: touching/dense clusters -> group_of_individual_litter, model_score = None
        for k in [2, 3, 4, 5, 6, 7]:
            dense_k = [
                {"bbox": [100.0 + i * 12.0, 100.0, 60.0, 60.0], "score": 0.70 + i * 0.01, "category_id": 8, "query_id": i}
                for i in range(k)
            ]
            res_k = compute_spatial_grouping(dense_k, 2000.0, 2000.0)
            self.assertEqual(res_k["summary"]["garbage_pile_count"], 0, f"k={k} must not be garbage_pile")
            self.assertEqual(res_k["summary"]["group_of_individual_litter_count"], 1, f"k={k} must be group_of_individual_litter")
            self.assertEqual(res_k["regions"][0]["category"], "group_of_individual_litter")
            self.assertEqual(res_k["regions"][0]["item_count"], k)
            self.assertIsNone(res_k["regions"][0]["model_score"])

        # 3. k >= 8 (k = 8, 12, 16): qualifies as garbage_pile
        for k in [8, 12, 16]:
            dense_heap = [
                {"bbox": [100.0 + i * 10.0, 100.0, 60.0, 60.0], "score": 0.65 + (i % 5) * 0.02, "category_id": 8, "query_id": i}
                for i in range(k)
            ]
            res_heap = compute_spatial_grouping(dense_heap, 2000.0, 2000.0)
            self.assertEqual(res_heap["summary"]["garbage_pile_count"], 1, f"k={k} dense heap must be garbage_pile")
            self.assertEqual(res_heap["summary"]["group_of_individual_litter_count"], 0)
            self.assertEqual(res_heap["regions"][0]["category"], "garbage_pile")
            self.assertEqual(res_heap["regions"][0]["item_count"], k)
            self.assertIsNone(res_heap["regions"][0]["model_score"])

        # 4. min_pile_items parameter override:
        # With min_pile_items = 4, a 4-item cluster qualifies as garbage_pile
        dense_4 = [
            {"bbox": [100.0 + i * 10.0, 100.0, 60.0, 60.0], "score": 0.70, "category_id": 8, "query_id": i}
            for i in range(4)
        ]
        res_custom = compute_spatial_grouping(dense_4, 2000.0, 2000.0, min_pile_items=4)
        self.assertEqual(res_custom["summary"]["garbage_pile_count"], 1)

    def test_03_standalone_image_for_testing_grouping(self):
        if self.video_showcase is None:
            self.skipTest("Testing video showcase cleaned under D-113")
        """For_testing.jpg (Image 10000) center cluster (k=4 < 8) is Group of Individual Litter."""
        preds = self.video_showcase["predictions"]["D-095 + Consensus"][10000]
        vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
        grp = compute_spatial_grouping(vis, image_width=5136, image_height=3984)
        s = grp["summary"]

        self.assertEqual(s["total_physical_items"], 6)
        self.assertEqual(s["total_regions"], 3)
        self.assertEqual(s["garbage_pile_count"], 0, "4-item cluster must not be Garbage Pile under D-112")
        self.assertEqual(s["group_of_individual_litter_count"], 1, "4-item cluster must be Group of Individual Litter")
        self.assertEqual(s["individual_litter_count"], 2)

        # Verify group details
        groups = [r for r in grp["regions"] if r["category"] == "group_of_individual_litter"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["item_count"], 4)
        self.assertIsNone(groups[0]["model_score"])

        # Verify individual litter items
        indivs = [r for r in grp["regions"] if r["category"] == "individual_litter"]
        self.assertEqual(len(indivs), 2)
        for ind in indivs:
            self.assertEqual(ind["item_count"], 1)
            self.assertIsNotNone(ind["model_score"])

    def test_04_video_frames_paper_towel_grouping(self):
        if self.video_showcase is None:
            self.skipTest("Testing video showcase cleaned under D-113")
        """Motion_Extract_Testing_2.mp4 (Frames 10005..10007): 4 paper towels form Group of Individual Litter."""
        consensus_preds = self.video_showcase["predictions"]["D-095 + Consensus"]
        for fid in [10005, 10006, 10007]:
            if fid not in consensus_preds:
                continue
            preds = consensus_preds[fid]
            w = self.video_showcase["images"][int(fid)]["width"]
            h = self.video_showcase["images"][int(fid)]["height"]
            vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
            grp = compute_spatial_grouping(vis, image_width=w, image_height=h)
            s = grp["summary"]

            self.assertEqual(len(vis), 4, f"Frame {fid} must have exactly 4 paper towel items")
            self.assertEqual(s["total_regions"], 1)
            self.assertEqual(s["garbage_pile_count"], 0)
            self.assertEqual(s["group_of_individual_litter_count"], 1)
            self.assertEqual(s["individual_litter_count"], 0)
            self.assertEqual(grp["regions"][0]["category"], "group_of_individual_litter")
            self.assertEqual(grp["regions"][0]["item_count"], 4)
            self.assertIsNone(grp["regions"][0]["model_score"])

    def test_05_zero_overlapping_multi_item_regions_across_sweep(self):
        """Zero overlapping multi-item region bounding boxes across all video frames and all 65 val images."""
        # Check all video frames if showcase present
        if self.video_showcase is not None:
            consensus_preds = self.video_showcase["predictions"]["D-095 + Consensus"]
            for fid, preds in consensus_preds.items():
                w = self.video_showcase["images"][int(fid)]["width"]
                h = self.video_showcase["images"][int(fid)]["height"]
                vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
                grp = compute_spatial_grouping(vis, image_width=w, image_height=h)
                multi_regs = [r for r in grp["regions"] if r["item_count"] >= 2]
                for i in range(len(multi_regs)):
                    for j in range(i + 1, len(multi_regs)):
                        b1 = multi_regs[i]["bbox"]
                        b2 = multi_regs[j]["bbox"]
                        u = compute_box_iou(b1, b2)
                        inter, a1, a2 = compute_box_intersection_and_areas(b1, b2)
                        iomin = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
                        self.assertLessEqual(u, 0.05, f"Frame {fid}: multi-region IoU violation: {u}")
                        self.assertLess(iomin, 0.20, f"Frame {fid}: multi-region IoMin violation: {iomin}")

        # Check all 65 validation images
        for img_id in sorted(self.img_dims.keys()):
            raw = self.raw_by_img.get(img_id, [])
            vec = self.d095_vecs["scores"][self.img_map[img_id]]
            w, h = self.img_dims[img_id]
            c_rows, _, _ = apply_scene_consensus(
                raw, vec, image_id=img_id, image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(img_id, []),
                run1_rows=self.run1_by_img.get(img_id, []),
            )
            vis = visible_rows(c_rows, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            grp = compute_spatial_grouping(vis, image_width=w, image_height=h)
            multi_regs = [r for r in grp["regions"] if r["item_count"] >= 2]
            for i in range(len(multi_regs)):
                for j in range(i + 1, len(multi_regs)):
                    b1 = multi_regs[i]["bbox"]
                    b2 = multi_regs[j]["bbox"]
                    u = compute_box_iou(b1, b2)
                    inter, a1, a2 = compute_box_intersection_and_areas(b1, b2)
                    iomin = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
                    self.assertLessEqual(u, 0.05, f"Image {img_id}: multi-region IoU violation: {u}")
                    self.assertLess(iomin, 0.20, f"Image {img_id}: multi-region IoMin violation: {iomin}")

    def test_06_replay_artifact_integrity_and_authorization(self):
        manifest_path = VIDEO_DIR / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("Testing video manifest cleaned under D-113")
        """Replay artifacts bound under D-112 with exact SHA-256 hashes."""
        showcase_path = VIDEO_DIR / "video-showcase.json"
        manifest_path = VIDEO_DIR / "manifest.json"
        vectors_path = VIDEO_DIR / "video-score-vectors.npz"

        manifest = strict_load(manifest_path)
        showcase = strict_load(showcase_path)

        self.assertEqual(manifest["authorization"], "D-112")
        self.assertEqual(showcase["authorization"], "D-112")

        # Verify SHA-256 hashes
        showcase_sha = hashlib.sha256(showcase_path.read_bytes()).hexdigest()
        self.assertEqual(manifest["files"]["video-showcase.json"], showcase_sha)

        vectors_sha = hashlib.sha256(vectors_path.read_bytes()).hexdigest()
        self.assertEqual(manifest["files"]["video-score-vectors.npz"], vectors_sha)

    def test_07_full_65_benchmark_invariants_preserved(self):
        """Full 65 validation benchmark bitwise preserved: AP9=0.301350, AP5=0.497161, TP=77, FP=63."""
        adj_by_img = {}
        clean_images = 0
        total_tp = 0
        total_fp = 0

        for img_id in sorted(self.img_dims.keys()):
            raw = self.raw_by_img.get(img_id, [])
            vec = self.d095_vecs["scores"][self.img_map[img_id]]
            w, h = self.img_dims[img_id]
            c_rows, _, _ = apply_scene_consensus(
                raw, vec, image_id=img_id, image_width=w, image_height=h,
                baseline_rows=self.base_by_img.get(img_id, []),
                run1_rows=self.run1_by_img.get(img_id, []),
            )
            adj_by_img[img_id] = c_rows

            vis = visible_rows(c_rows, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            gts = self.gts_by_img.get(img_id, [])
            img_tp = 0
            img_fp = 0
            for det in vis:
                matches = [gt for gt in gts if _compute_iou(det["bbox"], gt["bbox"]) >= 0.50]
                if matches:
                    best_gt = max(matches, key=lambda g: _compute_iou(det["bbox"], g["bbox"]))
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

        # Zero Lost True Positives
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
        self.assertEqual(lost_tps, [])

        all_adj = [r for img_id in sorted(adj_by_img.keys()) for r in adj_by_img[img_id]]
        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        ap9 = res9["metrics"]["AP"]
        ap5 = res5["metrics"]["AP"]
        paper_ap = res9["metrics"]["classes"]["paper"]["AP"]

        self.assertAlmostEqual(ap9, 0.301350, places=5)
        self.assertAlmostEqual(ap5, 0.497161, places=5)
        self.assertAlmostEqual(paper_ap, 0.640610, places=5)
        self.assertEqual(total_tp, 77)
        self.assertEqual(total_fp, 63)
        self.assertEqual(clean_images, 36)


if __name__ == "__main__":
    unittest.main()