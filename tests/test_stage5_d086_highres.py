"""Unit tests for Stage 5 High-Resolution (448/672) Workflow (D-086)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_d086_highres import (
    EXPECTED_D083_CHAMPION, EXPECTED_D083_E1, FROZEN_BACKBONE_HASH,
    D083_CHAMPION_BUNDLE, D083_E1_BUNDLE, D083_EXPORT
)
from scripts.taco_stage5_d083_core import _bundle_manifest_hashes


class TestStage5HighResD086(unittest.TestCase):

    def test_prerequisites_and_champion_hashes(self):
        # Read JSON manifest directly without reading forbidden private weight binary files during guarded suite
        champ_manifest = strict_load(D083_CHAMPION_BUNDLE / "soup_manifest.json")
        self.assertEqual(champ_manifest["parameter_sha256"], EXPECTED_D083_CHAMPION)

        e1_meta = strict_load(D083_E1_BUNDLE / "checkpoint_manifest.json")
        self.assertIn("files", e1_meta)

        d083_train = strict_load(D083_EXPORT / "train.json")
        self.assertEqual(len(d083_train["images"]), 379)
        self.assertEqual(len(d083_train["annotations"]), 731)

        d083_val = strict_load(D083_EXPORT / "validation.json")
        self.assertEqual(len(d083_val["images"]), 65)
        self.assertEqual(len(d083_val["annotations"]), 109)


if __name__ == "__main__":
    unittest.main()
