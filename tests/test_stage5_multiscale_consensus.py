"""Unit tests for Multi-Scale Query Fusion + Consensus Integration (D-088)."""
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from scripts.configuration import strict_load
from scripts.taco_stage5_multiscale_consensus import D086_RUN


class TestStage5MultiscaleConsensus(unittest.TestCase):

    def test_artifacts_exist_and_consistent(self):
        replay_path = D086_RUN / "multiscale-consensus-replay.json"
        scores_path = D086_RUN / "scores-multiscale-consensus.npz"
        audit_path = D086_RUN / "multiscale-consensus-audit.json"

        self.assertTrue(replay_path.is_file())
        self.assertTrue(scores_path.is_file())
        self.assertTrue(audit_path.is_file())

        replay = strict_load(replay_path)
        self.assertEqual(replay["authorization"], "D-088")
        self.assertEqual(replay["winning_pairing"], "D086_Soup2575_384_plus_D086_HR448")
        self.assertEqual(replay["winning_routing_policy"], "hr_small_medium_plus_can_paper")
        self.assertEqual(replay["winning_consensus_mode"], "pre_nms")
        self.assertAlmostEqual(replay["highest_9_specific_AP"], 0.258410, places=5)
        self.assertAlmostEqual(replay["highest_bantai_5_category_AP"], 0.430072, places=5)
        self.assertEqual(len(replay["raw_queries"]), 6500)

        with np.load(scores_path) as data:
            self.assertEqual(data["scores"].shape, (65, 100, 9))
            self.assertEqual(data["scores"].dtype, np.float32)
            self.assertEqual(len(data["image_ids"]), 65)

        audit = strict_load(audit_path)
        self.assertEqual(audit["authorization"], "D-088")
        self.assertIn("D086_Soup2575_384_plus_D086_HR448", audit["results"])
        self.assertIn("D083_Champion_384_plus_D086_HR448", audit["results"])
        self.assertIn("hr_small_medium_plus_can_paper", audit["results"]["D086_Soup2575_384_plus_D086_HR448"])


if __name__ == "__main__":
    unittest.main()
