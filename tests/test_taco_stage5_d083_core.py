"""CPU-only qualification for the isolated D-083 export and replay contracts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestD083Core(unittest.TestCase):
    def test_only_audited_core_siblings_are_selected(self):
        from scripts.prepare_stage5_d083_core_export import calculate_selection

        raw = {
            "categories": [
                {"id": 0, "name": "cigarette"},
                {"id": 2, "name": "plastic_bag"},
                {"id": 3, "name": "plastic_wrapper"},
                {"id": 5, "name": "drink_can"},
                {"id": 6, "name": "plastic_cup"},
                {"id": 7, "name": "styrofoam_piece"},
                {"id": 8, "name": "paper"},
                {"id": 9, "name": "Food Can"},
                {"id": 10, "name": "Aerosol"},
                {"id": 11, "name": "Plastified paper bag"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0},
                {"id": 2, "image_id": 2, "category_id": 9},
                {"id": 3, "image_id": 3, "category_id": 10},
                {"id": 4, "image_id": 4, "category_id": 11},
            ],
        }
        mapping = {"cigarette": 0, "plastic_bag": 2, "plastic_wrapper": 3,
                   "drink_can": 5, "plastic_cup": 6, "styrofoam_piece": 7,
                   "paper": 8, "Food Can": 5}
        selected, _, reasons = calculate_selection(raw, {1, 2, 3, 4}, {1}, mapping)
        self.assertEqual(selected, {1, 2})
        self.assertEqual(reasons[3], ["Aerosol"])
        self.assertEqual(reasons[4], ["Plastified paper bag"])

    def test_existing_supervised_labels_stay_nine_and_mapping_registry_extends(self):
        from scripts.taco_training_labels_v2 import LABELS
        from scripts.taco_stage5_material_hierarchy import OBJECT_REGISTRY, register_object_class

        self.assertEqual(len(LABELS), 9)
        self.assertEqual(tuple(OBJECT_REGISTRY[i]["specific_class"] for i in range(9)), LABELS)
        original = OBJECT_REGISTRY.get(10)
        try:
            record = register_object_class({
                "specific_category_id": 10,
                "specific_class": "audited_future_item",
                "bantai_category": "Organic",
                "material": "organic_biodegradable",
                "hazard_flag": False,
            })
            self.assertEqual(record["specific_class"], "audited_future_item")
            self.assertEqual(record["bantai_category"], "Organic")
            self.assertEqual(record["bantai_category_id"], 2)
        finally:
            if original is None:
                OBJECT_REGISTRY.pop(10, None)
            else:
                OBJECT_REGISTRY[10] = original

    def test_d083_run_discovery_requires_complete_evidence(self):
        from scripts.taco_stage5_visual_review import discover_d083_run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "runs/training/stage5-d083-core-incomplete"
            candidate.mkdir(parents=True)
            self.assertIsNone(discover_d083_run(root=root))
            self.assertEqual(discover_d083_run(explicit=candidate, root=root), candidate)

    def test_score_vector_artifact_requires_65_by_900_by_9(self):
        from scripts.taco_stage5_d083_core import save_vectors

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npz"
            invalid = {i: np.zeros((900, 8), dtype=np.float32) for i in range(65)}
            with self.assertRaises(ValueError):
                save_vectors(path, invalid)


if __name__ == "__main__":
    unittest.main()
