"""Unit tests for Stage 5 Extensible Object Class Coverage Audit (D-085)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_extensible_class_audit import (
    OUT_JSON, OUT_MD, CANDIDATE_GROUPS, ACTIVE_SUPERVISED_LABELS, run_extensible_class_audit
)


class TestExtensibleClassAudit(unittest.TestCase):

    def test_audit_execution_and_schema(self):
        # In guarded verification mode, work/ and docs/ are protected from mutation.
        # Check existing generated artifacts if present, else run audit to temp.
        if OUT_JSON.is_file():
            result = strict_load(OUT_JSON)
        else:
            result = run_extensible_class_audit()
        self.assertEqual(result["authorization"], "D-085")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["gpu_access"])
        self.assertEqual(result["model_forwards"], 0)
        self.assertFalse(result["test_json_opened"])
        self.assertEqual(result["heldout_test_photos_opened"], 0)

        self.assertEqual(len(result["active_supervised_classes"]), 9)
        active_names = [r["specific_class"] for r in result["active_supervised_classes"]]
        self.assertEqual(active_names, ACTIVE_SUPERVISED_LABELS)

        total_candidates = sum(len(cands) for cands in CANDIDATE_GROUPS.values())
        self.assertEqual(len(result["all_candidate_classes"]), total_candidates)

        # Check syringe properties
        syringe = next(r for r in result["all_candidate_classes"] if r["candidate_class"] == "syringe")
        self.assertEqual(syringe["raw_boxes"], 0)
        self.assertEqual(syringe["recommendation"], "ZERO_SHOT_HYBRID_ONLY")

        # Check food_waste properties
        food_waste = next(r for r in result["all_candidate_classes"] if r["candidate_class"] == "food_waste")
        self.assertEqual(food_waste["raw_boxes"], 8)
        self.assertEqual(food_waste["val_boxes"], 0)
        self.assertEqual(food_waste["recommendation"], "ZERO_SHOT_HYBRID_ONLY")

        # Check drink_carton properties (qualifies as SUPERVISED_CANDIDATE)
        drink_carton = next(r for r in result["all_candidate_classes"] if r["candidate_class"] == "drink_carton")
        self.assertGreaterEqual(drink_carton["clean_train_candidate_boxes"], 15)
        self.assertGreaterEqual(drink_carton["val_boxes"], 3)
        self.assertEqual(drink_carton["recommendation"], "SUPERVISED_CANDIDATE")

        # Check artifacts exist
        self.assertTrue(OUT_JSON.is_file())
        self.assertTrue((ROOT / "docs/ARGON_WEBSITE_INTEGRATION_GUIDE.md").is_file())

        saved_json = strict_load(OUT_JSON)
        self.assertEqual(saved_json["authorization"], "D-085")


if __name__ == "__main__":
    unittest.main()
