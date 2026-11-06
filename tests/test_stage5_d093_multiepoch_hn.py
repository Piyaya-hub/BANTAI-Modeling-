"""Unit tests for Stage 5 3-Epoch Hard-Negative & Class-Balanced Workflow (D-093)."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_d093_multiepoch_hn import (
    cosine_lr, mask_omitted_accessories, CLASS_WEIGHTS, CLASS_COUNTS, N_REF,
    D093_EXPORT, D093_RIGHTS, WARMSTART_BUNDLE, EXPECTED_WARMSTART,
    FROZEN_BACKBONE_HASH,
)


class TestStage5D093MultiepochHN(unittest.TestCase):

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
        lr_1 = cosine_lr(1, 1728, 1.8e-6, 2.0e-7)
        self.assertAlmostEqual(lr_1, 1.8e-6, places=12)

        lr_1728 = cosine_lr(1728, 1728, 1.8e-6, 2.0e-7)
        self.assertAlmostEqual(lr_1728, 2.0e-7, places=12)

        lrs = [cosine_lr(step, 1728, 1.8e-6, 2.0e-7) for step in range(1, 1729)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1])

    def test_absent_class_loss_logic(self):
        # Scenario 1: All classes present -> absent_classes is empty -> loss is 0.0
        prob_9 = torch.zeros((1, 900, 9), dtype=torch.float32)
        present_classes = set(range(9))
        absent_classes = [c for c in range(9) if c not in present_classes]
        self.assertEqual(len(absent_classes), 0)

        # Scenario 2: Absent classes with predictions <= 0.08 -> loss should be 0.0
        prob_9[:, :, :] = 0.05
        present_classes = {8}  # Only paper present
        absent_classes = [c for c in range(9) if c not in present_classes]
        loss_absent = sum(torch.relu(torch.topk(prob_9[0, :, c], k=10).values - 0.08).pow(2).mean()
                          for c in absent_classes) / len(absent_classes)
        self.assertAlmostEqual(float(loss_absent), 0.0, places=6)

        # Scenario 3: Absent class with false-positive query at 0.30 -> loss > 0.0
        prob_9[0, 0, 0] = 0.30  # false positive cigarette
        loss_absent_fp = sum(torch.relu(torch.topk(prob_9[0, :, c], k=10).values - 0.08).pow(2).mean()
                             for c in absent_classes) / len(absent_classes)
        self.assertGreater(float(loss_absent_fp), 0.0)
        expected_penalty = ((0.30 - 0.08) ** 2) / 10.0 / 8.0
        self.assertAlmostEqual(float(loss_absent_fp), expected_penalty, places=6)

    def test_export_and_rights_integrity(self):
        train = strict_load(D093_EXPORT / "train.json")
        self.assertEqual(len(train["images"]), 576)
        self.assertEqual(len(train["annotations"]), 1240)
        omitted_count = sum(len(img.get("omitted_accessory_bboxes", [])) for img in train["images"])
        self.assertEqual(omitted_count, 286)

        val = strict_load(D093_EXPORT / "validation.json")
        self.assertEqual(len(val["images"]), 65)
        self.assertEqual(len(val["annotations"]), 109)

        rights = strict_load(D093_RIGHTS)
        self.assertEqual(len(rights.get("photo_sha256", {})), 641)

    def test_warmstart_manifest_hashes(self):
        manifest = strict_load(WARMSTART_BUNDLE / "soup_manifest.json")
        self.assertEqual(manifest["parameter_sha256"], EXPECTED_WARMSTART)
        self.assertEqual(manifest["frozen_parameter_sha256"], FROZEN_BACKBONE_HASH)


if __name__ == "__main__":
    unittest.main()
