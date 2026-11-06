"""Unit tests for Stage 5 Tier 3 Core Fine-Tuning & Masking Workflow (D-089)."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_d089_tier3_core import (
    cosine_lr, mask_omitted_accessories, save_vectors,
    D089_EXPORT, D089_RIGHTS, D083_CHAMPION_BUNDLE, EXPECTED_D083_CHAMPION,
    D086_SOUP2575_BUNDLE, EXPECTED_D086_SOUP2575, FROZEN_BACKBONE_HASH,
)


class TestStage5D089Tier3Core(unittest.TestCase):

    def test_mask_omitted_accessories_no_omitted(self):
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        res = mask_omitted_accessories(img, [], [])
        self.assertIs(res, img)

    def test_mask_omitted_accessories_non_overlapping(self):
        # Create a 100x100 white image
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Put a black patch at (10, 10, 20, 20)
        arr[10:30, 10:30] = 0
        img = Image.fromarray(arr)
        mean_rgb = np.mean(arr, axis=(0, 1)).astype(np.uint8)

        # Omitted box at (10, 10, 20, 20), no GT boxes
        masked = mask_omitted_accessories(img, [[10.0, 10.0, 20.0, 20.0]], [])
        res_arr = np.array(masked)

        # Check that pixels inside the omitted box were set to mean RGB
        np.testing.assert_array_equal(res_arr[15, 15], mean_rgb)
        # Check that pixels outside were untouched
        np.testing.assert_array_equal(res_arr[50, 50], [200, 200, 200])

    def test_mask_omitted_accessories_preserves_overlapping_ground_truth(self):
        # Create a 100x100 image
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Put red in (10, 10, 20, 20)
        arr[10:30, 10:30] = [255, 0, 0]
        img = Image.fromarray(arr)

        # Omitted box at (10, 10, 20, 20), but GT box covers (10, 10, 20, 20)
        masked = mask_omitted_accessories(img, [[10.0, 10.0, 20.0, 20.0]], [[10.0, 10.0, 20.0, 20.0]])
        res_arr = np.array(masked)

        # Pixels inside GT must NOT be masked!
        np.testing.assert_array_equal(res_arr[15, 15], [255, 0, 0])

    def test_cosine_lr_monotonic_and_endpoints(self):
        lr_1 = cosine_lr(1, 576, 1.5e-6, 2.0e-7)
        self.assertAlmostEqual(lr_1, 1.5e-6, places=12)

        lr_576 = cosine_lr(576, 576, 1.5e-6, 2.0e-7)
        self.assertAlmostEqual(lr_576, 2.0e-7, places=12)

        # Check monotonic decrease
        lrs = [cosine_lr(step, 576, 1.5e-6, 2.0e-7) for step in range(1, 577)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1])

    def test_export_and_rights_integrity(self):
        train = strict_load(D089_EXPORT / "train.json")
        self.assertEqual(len(train["images"]), 576)
        self.assertEqual(len(train["annotations"]), 1240)
        omitted_count = sum(len(img.get("omitted_accessory_bboxes", [])) for img in train["images"])
        self.assertEqual(omitted_count, 286)

        val = strict_load(D089_EXPORT / "validation.json")
        self.assertEqual(len(val["images"]), 65)
        self.assertEqual(len(val["annotations"]), 109)

        rights = strict_load(D089_RIGHTS)
        self.assertEqual(len(rights.get("photo_sha256", {})), 641)

    def test_checkpoint_manifest_hashes(self):
        d083_manifest = strict_load(D083_CHAMPION_BUNDLE / "soup_manifest.json")
        self.assertEqual(d083_manifest["parameter_sha256"], EXPECTED_D083_CHAMPION)
        self.assertEqual(d083_manifest["frozen_parameter_sha256"], FROZEN_BACKBONE_HASH)

        d086_manifest = strict_load(D086_SOUP2575_BUNDLE / "soup_manifest.json")
        self.assertEqual(d086_manifest["parameter_sha256"], EXPECTED_D086_SOUP2575)

    def test_save_vectors_requires_65_by_900_by_9(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.npz"
            invalid_shape = {i: np.zeros((900, 8), dtype=np.float32) for i in range(65)}
            with self.assertRaises(ValueError):
                save_vectors(path, invalid_shape)


if __name__ == "__main__":
    unittest.main()
