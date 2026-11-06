import unittest

from scripts.taco_stage5_consensus_tiebreak import apply_consensus_tiebreak


class TestScorePreservingConsensus(unittest.TestCase):
    def test_paper_to_plastic_film_preserves_score_and_geometry(self):
        row = {"query_id": 0, "category_id": 8, "score": 0.3078179955482483,
               "bbox": [10, 20, 30, 40]}
        scores = [.1130877063, .2749033570, .1952677816, .2554721832,
                  .2927520871, .2038989961, .1313864738, .1276722550,
                  .3078179955]
        changed, reassigned = apply_consensus_tiebreak(
            row, scores, delta=.016,
            coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score")
        self.assertTrue(reassigned)
        self.assertEqual(changed["category_id"], 4)
        self.assertEqual(changed["score"], row["score"])
        self.assertEqual(changed["bbox"], row["bbox"])
        self.assertEqual(row["category_id"], 8)

    def test_thresholds_and_coactivation_are_required(self):
        row = {"query_id": 0, "category_id": 8, "score": .31, "bbox": [0, 0, 2, 2]}
        scores = [.1, .28, .1, .1, .29, .1, .1, .1, .30]
        for kwargs in (
            {"delta": .009, "coactivation": "2 plastic subtypes >= 0.25"},
            {"delta": .016, "coactivation": "3 plastic subtypes >= 0.20"},
        ):
            result, changed = apply_consensus_tiebreak(row, scores, **kwargs)
            self.assertFalse(changed)
            self.assertEqual(result["category_id"], 8)
        below_paper = scores.copy()
        below_paper[8] = .149
        self.assertFalse(apply_consensus_tiebreak(row, below_paper)[1])

    def test_gui_has_reversible_saved_vector_toggle_before_mapping(self):
        import inspect
        from scripts.taco_stage5_visual_review import Stage5ViewerGUI
        source = inspect.getsource(Stage5ViewerGUI)
        self.assertIn("Score-preserving plastic consensus", source)
        self.assertIn("apply_to_query_rows", source)
        self.assertIn('scheme == "bantai-5-category"', source)
        self.assertIn("self.taxonomy_var.get() != VIEWS[2]", source)


if __name__ == "__main__":
    unittest.main()
