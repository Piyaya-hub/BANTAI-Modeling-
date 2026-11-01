"""Unit tests for Stage 5 Class-Balanced & L2-Anchored Tier 3 Core Workflow (D-091)."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_d091_balanced_tier3 import (
    cosine_lr, mask_omitted_accessories, CLASS_WEIGHTS, CLASS_COUNTS, N_REF,
    D091_EXPORT, D091_RIGHTS, D086_SOUP2575_BUNDLE, EXPECTED_D086_SOUP2575,
    D083_CHAMPION_BUNDLE, EXPECTED_D083_CHAMPION, D089_CHAMPION_BUNDLE, EXPECTED_D089_CHAMPION,
    FROZEN_BACKBONE_HASH,
)


class TestStage5D091BalancedTier3(unittest.TestCase):

    def test_class_weights_inverse_sqrt(self):
        self.assertEqual(len(CLASS_WEIGHTS), 9)
        # Verify rare classes get higher weight than over-represented classes
        self.assertGreater(CLASS_WEIGHTS[2], CLASS_WEIGHTS[0])  # plastic_bag > cigarette
        self.assertGreater(CLASS_WEIGHTS[7], CLASS_WEIGHTS[8])  # styrofoam > paper
        self.assertGreater(CLASS_WEIGHTS[6], CLASS_WEIGHTS[1])  # plastic_cup > bottle
        # Check clipping bounds [0.75, 1.80]
        for c, w in CLASS_WEIGHTS.items():
            self.assertGreaterEqual(w, 0.75)
            self.assertLessEqual(w, 1.80)
            expected_raw = math.sqrt(N_REF / CLASS_COUNTS[c])
            expected_clipped = min(1.80, max(0.75, expected_raw))
            self.assertAlmostEqual(w, expected_clipped, places=6)

    def test_mask_omitted_accessories_no_omitted(self):
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        res = mask_omitted_accessories(img, [], [])
        self.assertIs(res, img)

    def test_mask_omitted_accessories_non_overlapping(self):
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        arr[10:30, 10:30] = 0
        img = Image.fromarray(arr)
        mean_rgb = np.mean(arr, axis=(0, 1)).astype(np.uint8)

        masked = mask_omitted_accessories(img, [[10.0, 10.0, 20.0, 20.0]], [])
        res_arr = np.array(masked)

        np.testing.assert_array_equal(res_arr[15, 15], mean_rgb)
        np.testing.assert_array_equal(res_arr[50, 50], [200, 200, 200])

    def test_mask_omitted_accessories_preserves_overlapping_ground_truth(self):
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        arr[10:30, 10:30] = [255, 0, 0]
        img = Image.fromarray(arr)

        masked = mask_omitted_accessories(img, [[10.0, 10.0, 20.0, 20.0]], [[10.0, 10.0, 20.0, 20.0]])
        res_arr = np.array(masked)

        np.testing.assert_array_equal(res_arr[15, 15], [255, 0, 0])

    def test_cosine_lr_monotonic_and_endpoints(self):
        lr_1 = cosine_lr(1, 576, 8.0e-7, 1.5e-7)
        self.assertAlmostEqual(lr_1, 8.0e-7, places=12)

        lr_576 = cosine_lr(576, 576, 8.0e-7, 1.5e-7)
        self.assertAlmostEqual(lr_576, 1.5e-7, places=12)

        lrs = [cosine_lr(step, 576, 8.0e-7, 1.5e-7) for step in range(1, 577)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1])

    def test_export_and_rights_integrity(self):
        train = strict_load(D091_EXPORT / "train.json")
        self.assertEqual(len(train["images"]), 576)
        self.assertEqual(len(train["annotations"]), 1240)
        omitted_count = sum(len(img.get("omitted_accessory_bboxes", [])) for img in train["images"])
        self.assertEqual(omitted_count, 286)

        val = strict_load(D091_EXPORT / "validation.json")
        self.assertEqual(len(val["images"]), 65)
        self.assertEqual(len(val["annotations"]), 109)

        rights = strict_load(D091_RIGHTS)
        self.assertEqual(len(rights.get("photo_sha256", {})), 641)

    def test_checkpoint_manifest_hashes(self):
        d086_manifest = strict_load(D086_SOUP2575_BUNDLE / "soup_manifest.json")
        self.assertEqual(d086_manifest["parameter_sha256"], EXPECTED_D086_SOUP2575)
        self.assertEqual(d086_manifest["frozen_parameter_sha256"], FROZEN_BACKBONE_HASH)

        d083_manifest = strict_load(D083_CHAMPION_BUNDLE / "soup_manifest.json")
        self.assertEqual(d083_manifest["parameter_sha256"], EXPECTED_D083_CHAMPION)

        d089_manifest = strict_load(D089_CHAMPION_BUNDLE / "soup_manifest.json")
        self.assertEqual(d089_manifest["parameter_sha256"], EXPECTED_D089_CHAMPION)


if __name__ == "__main__":
    unittest.main()
