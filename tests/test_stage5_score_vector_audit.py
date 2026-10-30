import unittest

import numpy as np

from scripts.taco_stage5_score_vector_audit import aggregate
from scripts.taco_stage5_material_hierarchy import NAMES


class ScoreVectorPoolingTests(unittest.TestCase):
    def test_max_subtype_matches_argmax_material_probability(self):
        # paper wins over each individual plastic subtype
        row = np.asarray([[.01, .20, .10, .18, .19, .02, .03, .01, .21]], dtype=np.float32)
        for scheme in ("5-material", "4-material"):
            pooled = aggregate(row, scheme, "max_subtype")[0]
            self.assertEqual(NAMES[scheme][int(pooled.argmax())], "paper_waste" if scheme == "5-material" else "paper_waste")

    def test_noisy_or_can_flip_material_without_changing_specific_scores(self):
        row = np.asarray([[.01, .20, .10, .18, .19, .02, .03, .01, .21]], dtype=np.float32)
        for scheme, plastic_name in (("5-material", "plastic_waste"),
                                     ("4-material", "plastic_polymer_waste")):
            pooled = aggregate(row, scheme, "noisy_or")[0]
            self.assertEqual(NAMES[scheme][int(pooled.argmax())], plastic_name)
        self.assertAlmostEqual(float(row[0, 8]), .21, places=6)

    def test_bounded_top_two_is_capped(self):
        row = np.ones((1, 9), dtype=np.float32)
        self.assertTrue(np.all(aggregate(row, "5-material", "bounded_top2_sum") <= 1.0))
        self.assertTrue(np.all(aggregate(row, "4-material", "bounded_top2_sum") <= 1.0))


if __name__ == "__main__":
    unittest.main()
