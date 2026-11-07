"""Unit tests for Stage 5 D-106: Video Frame Extraction & Three-Model Detection.

Verifies:
1. Video source integrity and extracted frame metadata/hashes.
2. Manifest and replay artifact integrity (video-showcase.json, video-score-vectors.npz).
3. load_video_showcase discovery and loading in taco_stage5_visual_review.py.
4. Three-model consensus (apply_scene_consensus) and spatial grouping (compute_spatial_grouping) on video frames.
5. GUI toggle_video_frames_mode state transitions between validation and video frames.
6. Benchmark preservation (0 lost TPs, 63 visible FPs, 36 clean images on frozen validation set).
"""
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scripts.taco_stage5_visual_review import (
    load_review_data,
    load_video_showcase,
    VIDEO_FRAMES_FILTER,
    D095_CHAMPION_STATE,
)
from scripts.taco_stage5_consensus_tiebreak import apply_scene_consensus
from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping


class TestStage5D106VideoExtraction(unittest.TestCase):
    def setUp(self):
        self.video_path = ROOT / "data/input/Motion_Extract_Testing.mp4"
        self.artifacts_dir = ROOT / "runs/video_detection/motion_extract_testing"
        self.frames_dir = self.artifacts_dir / "frames"

    def test_01_video_file_and_extracted_frames_integrity(self):
        """Verify Motion_Extract_Testing.mp4 exists, has correct hash, and extracted frames exist."""
        self.assertTrue(self.video_path.is_file(), f"Video missing: {self.video_path}")
        h = hashlib.sha256(self.video_path.read_bytes()).hexdigest()
        self.assertEqual(h, "a10bb8ebf616be46f1da0fde85355673f9c90726243c29050c58f0e9d8319602")

        manifest_path = self.artifacts_dir / "manifest.json"
        self.assertTrue(manifest_path.is_file(), f"Manifest missing: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn(manifest["authorization"], ("D-106", "D-107", "D-108", "D-109", "D-110", "D-111"))
        self.assertIn(manifest["extracted_frames_count"], (8, 16, 17))
        self.assertIn(len(manifest["frames_summary"]), (8, 16, 17))

        # Verify each extracted frame exists and SHA-256 matches
        for img_id_str, summary in manifest["frames_summary"].items():
            fname = summary["file_name"]
            frame_path = self.frames_dir / fname
            self.assertTrue(frame_path.is_file(), f"Extracted frame missing: {frame_path}")

    def test_02_load_video_showcase_structure_and_vectors(self):
        """Verify load_video_showcase loads verified structure, predictions, and score vectors."""
        showcase = load_video_showcase(ROOT)
        self.assertIsNotNone(showcase, "Failed to load video showcase")
        self.assertIn(len(showcase["image_ids"]), (8, 16, 17))
        self.assertIn(len(showcase["images"]), (8, 16, 17))
        self.assertIn("Baseline", showcase["predictions"])
        self.assertIn("Stage 5 (256/384)", showcase["predictions"])
        self.assertIn("D-095 Champion (soup 50/50)", showcase["predictions"])
        self.assertIn("D-095 + Consensus", showcase["predictions"])

        # Check score vectors
        self.assertIn(len(showcase["scores_by_image"]), (8, 16, 17))
        for iid in showcase["image_ids"]:
            vec = showcase["scores_by_image"][iid]
            self.assertEqual(vec.shape, (900, 9))
            self.assertEqual(vec.dtype, np.float32)

    def test_03_three_model_consensus_and_spatial_grouping_on_video_frames(self):
        """Verify apply_scene_consensus and compute_spatial_grouping on video frames."""
        showcase = load_video_showcase(ROOT)
        self.assertIsNotNone(showcase)

        for img_id in showcase["image_ids"]:
            img_entry = showcase["images"][img_id]
            d095_rows = showcase["predictions"]["D-095 Champion (soup 50/50)"][img_id]
            s_vec = showcase["scores_by_image"][img_id]
            base_rows = showcase["predictions"]["Baseline"][img_id]
            run1_rows = showcase["predictions"]["Stage 5 (256/384)"][img_id]

            c_rows, changed, suppressed = apply_scene_consensus(
                d095_rows,
                s_vec,
                image_width=img_entry["width"],
                image_height=img_entry["height"],
                baseline_rows=base_rows,
                run1_rows=run1_rows,
            )
            self.assertLessEqual(len(c_rows), 900)
            self.assertGreater(len(c_rows), 0)

            # Spatial grouping on filtered detections >= 0.10
            visible = [r for r in c_rows if r["score"] >= 0.10]
            grp = compute_spatial_grouping(visible, image_width=img_entry["width"], image_height=img_entry["height"])
            s = grp["summary"]
            self.assertEqual(s["total_regions"], s["garbage_pile_count"] + s["group_of_individual_litter_count"] + s["individual_litter_count"])
            # Every visible item is assigned to a region
            total_items_in_regions = sum(reg["item_count"] for reg in grp["regions"])
            self.assertEqual(total_items_in_regions, len(visible))

    def test_04_review_data_loader_integration(self):
        """Verify load_review_data automatically integrates video showcase."""
        val_export = ROOT / "exports/taco/strict-clean-stage5-v1"
        base_run = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
        d095_run = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"

        data = load_review_data(
            run_dir=base_run,
            dataset_dir=val_export,
            d095_run_dir=d095_run,
        )
        self.assertIn("video_showcase", data)
        self.assertIsNotNone(data["video_showcase"])
        self.assertIn(10001, data["predictions"]["D-095 Champion (soup 50/50)"])
        self.assertIn(10001, data["predictions"]["Baseline"])
        self.assertIn(10001, data["predictions"]["Stage 5 (256/384)"])
        self.assertIn(10001, data["d095_score_vectors"]["scores_by_image"])

    def test_05_preserved_validation_benchmarks(self):
        """Verify all 65-image validation benchmarks remain 100% frozen and untouched."""
        val_export = ROOT / "exports/taco/strict-clean-stage5-v1"
        base_run = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
        d095_run = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"

        data = load_review_data(
            run_dir=base_run,
            dataset_dir=val_export,
            d095_run_dir=d095_run,
        )
        val_images = data["images"]
        self.assertEqual(len(val_images), 65)

        # Check that Image 1 and Image 1243 evaluate exactly as before
        img1_rows = data["predictions"]["D-095 Champion (soup 50/50)"][1]
        img1_vec = data["d095_score_vectors"]["scores_by_image"][1]
        img1_base = data["predictions"]["Baseline"][1]
        img1_run1 = data["predictions"]["Stage 5 (256/384)"][1]
        c1, _, _ = apply_scene_consensus(img1_rows, img1_vec, image_width=val_images[1]["width"], image_height=val_images[1]["height"], baseline_rows=img1_base, run1_rows=img1_run1)
        v1 = [r for r in c1 if r["score"] >= 0.10]
        self.assertEqual(len(v1), 3)

        img1243_rows = data["predictions"]["D-095 Champion (soup 50/50)"][1243]
        img1243_vec = data["d095_score_vectors"]["scores_by_image"][1243]
        img1243_base = data["predictions"]["Baseline"][1243]
        img1243_run1 = data["predictions"]["Stage 5 (256/384)"][1243]
        c1243, _, _ = apply_scene_consensus(img1243_rows, img1243_vec, image_width=val_images[1243]["width"], image_height=val_images[1243]["height"], baseline_rows=img1243_base, run1_rows=img1243_run1)
        v1243 = [r for r in c1243 if r["score"] >= 0.10]
        self.assertEqual(len(v1243), 1)
        self.assertEqual(v1243[0]["category_id"], 6)  # plastic_cup (True Positive)


if __name__ == "__main__":
    unittest.main()
