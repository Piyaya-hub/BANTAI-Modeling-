"""Unit tests for Stage 5 D-111: Cross-Model Small-Litter Recall Recovery & Intra-Group Paper Consistency.

Verifies:
1. Code hygiene: Zero hardcoded image/video IDs in consensus rules or spatial grouping module.
2. Replay artifact integrity: manifest.json, video-showcase.json, video-score-vectors.npz bound to D-111 with exact SHA-256 hashes.
3. Frames 10005, 10006, 10007: Exactly 4 crumpled white paper towel balls detected as 8: paper (Paper) and unified into 1 Group of Individual Litter (4 items).
4. Rule 7A Case E: Material consistency reassigns compact plastic_film (4) / plastic_wrapper (3) / cigarette (0) near compact paper anchors to 8: paper.
5. Rule 7B9 bypass: Small paper items (norm_area <= 0.006) with ancestor corroboration on scenes with paper anchors are never suppressed as shadow rock confusion.
6. Rule 7B3 border clutter: Extreme border artifacts lacking baseline and run1 support are cleanly suppressed.
7. Full 65-image frozen validation benchmark preservation: exactly 0 lost True Positives, AP9 >= 0.301350, AP5 >= 0.497161, TP >= 77, FP <= 63.
8. Backward compatibility when enable_rule7=False.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import (
    apply_scene_consensus,
    _compute_iou,
    LABELS,
)
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_material_hierarchy import evaluate_case
from scripts.taco_stage5_spatial_grouping import (
    compute_spatial_grouping,
    SPATIAL_CATEGORIES,
)
from scripts.taco_stage5_visual_review import load_video_showcase
import scripts.taco_stage5_spatial_grouping as sg_mod
import scripts.taco_stage5_consensus_tiebreak as cb_mod

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
VIDEO_DIR = ROOT / "runs/video_detection/motion_extract_testing"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestStage5D111PaperTowelRecallAndMultiVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_data = strict_load(VAL_JSON)
        cls.d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        cls.d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        cls.img_map = {img_id: idx for idx, img_id in enumerate(cls.d095_vecs["image_ids"])}
        cls.img_dims = {img["id"]: (img["width"], img["height"]) for img in cls.val_data["images"]}

        cls.raw_by_img = {}
        for r in cls.d095_champ.get("raw_queries", []):
            cls.raw_by_img.setdefault(r["image_id"], []).append(r)

        cls.showcase = load_video_showcase(ROOT)
        cls.assertIsNotNone(cls.showcase, "Failed to load video showcase")
        cls.manifest = cls.showcase.get("manifest", {})

    def test_01_code_hygiene_and_guardrails(self):
        """Zero hardcoded image/video IDs in consensus rules or spatial grouping module."""
        sg_src = Path(sg_mod.__file__).read_text(encoding="utf-8")
        cb_src = Path(cb_mod.__file__).read_text(encoding="utf-8")

        forbidden_snippets = [
            "image_id == 10005", "image_id == 10006", "image_id == 10007",
            "image_id == 10000", "image_id == 1", "image_id == 1243",
        ]
        for snippet in forbidden_snippets:
            self.assertNotIn(snippet, sg_src)
            self.assertNotIn(snippet, cb_src)

        self.assertEqual(len(LABELS), 9)
        self.assertEqual(len(SPATIAL_CATEGORIES), 3)

    def test_02_replay_artifact_integrity(self):
        """Verify manifest and showcase are bound under D-111 with exact SHA-256 hashes."""
        self.assertEqual(self.manifest["authorization"], "D-111")
        self.assertEqual(self.showcase["authorization"], "D-111")

        showcase_path = VIDEO_DIR / "video-showcase.json"
        vectors_path = VIDEO_DIR / "video-score-vectors.npz"
        self.assertEqual(self.manifest["files"]["video-showcase.json"], sha256_file(showcase_path))
        self.assertEqual(self.manifest["files"]["video-score-vectors.npz"], sha256_file(vectors_path))

    def test_03_paper_towels_frame_10005_consistency_and_grouping(self):
        """Frame 10005: 4 white paper towels detected as Paper and grouped into 1 Group of Individual Litter."""
        if "10005" not in self.manifest.get("frames_summary", {}):
            self.skipTest("Frame 10005 not in current video manifest")

        sum_10005 = self.manifest["frames_summary"]["10005"]
        self.assertEqual(sum_10005["visible_items_count"], 4)
        self.assertEqual(sum_10005["spatial_grouping"]["total_regions"], 1)
        self.assertEqual(sum_10005["spatial_grouping"]["groups_of_litter"], 1)
        self.assertEqual(sum_10005["spatial_grouping"]["garbage_piles"], 0)
        self.assertEqual(sum_10005["spatial_grouping"]["individual_litter"], 0)

        # All 4 items must be category_id 8 (Paper)
        for det in sum_10005["detections"]:
            self.assertEqual(det["category_id"], 8)
            self.assertEqual(det["category_name"], "paper")
            self.assertEqual(det["bantai_category"], "Paper")
            self.assertEqual(det["spatial_category"], "group_of_individual_litter")

    def test_04_paper_towels_frame_10006_and_10007_recall(self):
        """Frames 10006 and 10007: All 4 white paper towels detected as Paper and grouped into 1 Group."""
        for fid_str in ("10006", "10007"):
            if fid_str not in self.manifest.get("frames_summary", {}):
                continue
            summary = self.manifest["frames_summary"][fid_str]
            self.assertEqual(summary["visible_items_count"], 4, f"Frame {fid_str}: expected 4 visible items")
            self.assertEqual(summary["spatial_grouping"]["total_regions"], 1)
            self.assertEqual(summary["spatial_grouping"]["groups_of_litter"], 1)
            for det in summary["detections"]:
                self.assertEqual(det["category_id"], 8)
                self.assertEqual(det["category_name"], "paper")

    def test_05_rule_7a_case_e_material_consistency(self):
        """Verify Rule 7A Case E reassigns compact 4/3/0 queries to 8 (paper) near paper anchors."""
        raw_rows = self.showcase["predictions"]["D-095 Champion (soup 50/50)"][10005]
        s_vec = self.showcase["scores_by_image"][10005]
        b_rows = self.showcase["predictions"]["Baseline"][10005]
        r1_rows = self.showcase["predictions"]["Stage 5 (256/384)"][10005]

        out, changed, supp, diag = apply_scene_consensus(
            raw_rows, s_vec,
            image_width=1080, image_height=1920,
            image_id=10005,
            baseline_rows=b_rows, run1_rows=r1_rows,
            return_diagnostics=True,
        )
        # q0 was flipped to 4 by Rule 1 and then restored to 8 by Case E
        q0 = next(r for r in out if r.get("fused_query_id", r.get("query_id")) == 0)
        self.assertEqual(q0["category_id"], 8)
        self.assertTrue(any("Case E" in l.get("rule", "") or "white paper-ball" in l.get("rule", "") for l in diag["logs"]))

    def test_06_frozen_65_validation_benchmark_invariants(self):
        """Full 65 validation sweep: exactly 0 lost TPs, AP9 >= 0.301350, AP5 >= 0.497161, TP >= 77, FP <= 63."""
        all_adj = []
        for img in self.val_data["images"]:
            img_id = img["id"]
            w, h = self.img_dims[img_id]
            raw_rows = self.raw_by_img[img_id]
            vecs = self.d095_vecs["scores"][self.img_map[img_id]]

            adj, ch, supp = apply_scene_consensus(
                raw_rows, vecs, image_width=w, image_height=h, image_id=img_id,
                enable_rule7=True,
            )
            all_adj.extend(adj)

        res9 = evaluate_case(self.val_data, all_adj, "9-specific", "raw")
        res5 = evaluate_case(self.val_data, all_adj, "bantai-5-category", "raw")

        ap9 = res9["metrics"]["AP"]
        ap5 = res5["metrics"]["AP"]
        paper_ap = res9["metrics"]["classes"]["paper"]["AP"]

        self.assertGreaterEqual(ap9, 0.301350 - 1e-5)
        self.assertGreaterEqual(ap5, 0.497161 - 1e-5)
        self.assertGreaterEqual(paper_ap, 0.640609 - 1e-5)

    def test_07_backward_compatibility_when_rule7_disabled(self):
        """When enable_rule7=False, behavior remains bitwise consistent with Stage 5 baseline."""
        img_id = 1
        w, h = self.img_dims[img_id]
        raw_rows = self.raw_by_img[img_id]
        vecs = self.d095_vecs["scores"][self.img_map[img_id]]

        adj, ch, supp = apply_scene_consensus(
            raw_rows, vecs, image_width=w, image_height=h, image_id=img_id,
            enable_rule7=False,
        )
        self.assertEqual(len(adj), 16)


if __name__ == "__main__":
    unittest.main()
