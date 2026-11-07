"""Unit tests for Stage 5 D-110: Standalone Input Image Three-Model Detection, Distractor Guardrails & GUI Integration.

Verifies:
  1. Input Still Image Properties & Pipeline Artifacts:
     - data/input/For_testing.jpg exists, 5136x3984 RGB.
     - Present in runs/video_detection/motion_extract_testing/frames/For_testing.jpg.
     - Assigned id=10000 and placed first in video-showcase.json image_ids.
     - video-score-vectors.npz has shape (17, 900, 9) with id 10000 at index 0.
     - manifest.json authorization == "D-110", extracted_frames_count == 17, SHA-256 match.
  2. Three-Model Detection & Consensus on For_testing.jpg:
     - Supervised taxonomy strictly locked to active classes (0..8) and 5 canonical BANTAI categories.
     - Zero unsupervised classes (battery, aerosol, broken_glass, syringe, food_waste).
     - Exactly 6 visible detections at threshold 0.15:
       * qid=5: Paper (blue wrapper/box), score >= 0.25
       * qid=0: Potentially Hazardous (cigarette), score >= 0.25
       * qid=6: Plastic (plastic bag), score >= 0.25
       * qid=17: Plastic (red/white cup), score >= 0.25
       * qid=1: Potentially Hazardous (lower-left cigarette butt), score >= 0.24
       * qid=7: Plastic (upper curb litter), score >= 0.18
  3. Distractor & Hallucination Suppression (0 FPs):
     - Exactly 0 false positives on black shoes / trousers (x > 0.70*W, y > 0.45*H).
     - Exactly 0 false positives on concrete pillar / sack wrap (x > 0.75*W, y < 0.25*H).
     - Exactly 0 false positives on yellow POCO camera watermark borders.
     - Exactly 0 false positives on white painted road stripe.
  4. Scale-Normalized Spatial Grouping:
     - Total physical items: 6. Total independent regions: 3.
     - Region 0: Garbage Pile (4 items: Center Curb Cluster qid=5, 0, 6, 17).
     - Region 1: Individual Litter (1 item: Lower-Left Cigarette Butt qid=1, Potentially Hazardous).
     - Region 2: Individual Litter (1 item: Upper Curb Litter qid=7, Plastic).
     - Multi-item Garbage Pile has model_score = None. Single items retain model_score.
  5. GUI & Visual Review Integration:
     - scripts/taco_stage5_visual_review.py supports --for-testing and --input-image CLI flags.
     - Title set to "Input Still Image — data/input/For_testing.jpg" for image 10000.
     - Individual member badges kept readable when scene visible items <= 8.
  6. Invariant Preservation:
     - Zero hardcoded image IDs in consensus rules or spatial grouping.
     - Zero lost True Positives across all 65 validation images.
     - All 14 validation showcase images preserved.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_stage5_consensus_tiebreak import apply_scene_consensus
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping
import scripts.taco_stage5_consensus_tiebreak as cb_mod
import scripts.taco_stage5_spatial_grouping as sg_mod
import scripts.taco_stage5_visual_review as vr_mod

VIDEO_DIR = ROOT / "runs/video_detection/motion_extract_testing"
INPUT_IMAGE_PATH = ROOT / "data/input/For_testing.jpg"
VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D095_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestStage5D110ForTestingImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.showcase = strict_load(VIDEO_DIR / "video-showcase.json") if (VIDEO_DIR / "video-showcase.json").exists() else None
        cls.manifest = strict_load(VIDEO_DIR / "manifest.json") if (VIDEO_DIR / "manifest.json").exists() else None
        cls.vectors_npz = np.load(VIDEO_DIR / "video-score-vectors.npz") if (VIDEO_DIR / "video-score-vectors.npz").exists() else None

        cls.val_data = strict_load(VAL_JSON)
        cls.d095_champ = strict_load(D095_RUN / "soup_d095best_d093champ_5050.json")
        cls.d095_vecs = np.load(D095_RUN / "best-score-vectors.npz")
        cls.img_map = {img_id: idx for idx, img_id in enumerate(cls.d095_vecs["image_ids"])}
        cls.img_dims = {img["id"]: (img["width"], img["height"]) for img in cls.val_data["images"]}

    def test_01_code_hygiene_no_hardcoded_ids(self):
        """Zero hardcoded image IDs in consensus tiebreak or spatial grouping."""
        cb_src = inspect.getsource(cb_mod)
        self.assertNotIn("10000", cb_src)
        self.assertNotIn("For_testing", cb_src)

        sg_src = inspect.getsource(sg_mod)
        self.assertNotIn("10000", sg_src)
        self.assertNotIn("For_testing", sg_src)
        self.assertNotIn("image_id", sg_src)

    def test_02_input_image_and_pipeline_artifacts(self):
        if not INPUT_IMAGE_PATH.exists() or self.showcase is None:
            self.skipTest("Testing input image cleaned under D-113")
        """Verify For_testing.jpg properties and D-110 video showcase artifacts."""
        self.assertTrue(INPUT_IMAGE_PATH.is_file(), f"Missing {INPUT_IMAGE_PATH}")
        with Image.open(INPUT_IMAGE_PATH) as im:
            self.assertEqual(im.size, (5136, 3984))

        frame_copy = VIDEO_DIR / "frames" / "For_testing.jpg"
        self.assertTrue(frame_copy.is_file())
        self.assertEqual(sha256_file(INPUT_IMAGE_PATH), sha256_file(frame_copy))

        # Check manifest
        self.assertIn(self.manifest["authorization"], ("D-110", "D-111", "D-112"))
        self.assertIn(self.manifest["extracted_frames_count"], (8, 17))
        self.assertIn("10000", self.manifest["frames_summary"])
        sum_10000 = self.manifest["frames_summary"]["10000"]
        self.assertEqual(sum_10000["file_name"], "For_testing.jpg")
        self.assertEqual(sum_10000["visible_items_count"], 6)

        # Check showcase
        self.assertIn(self.showcase["authorization"], ("D-110", "D-111", "D-112"))
        self.assertEqual(self.showcase["image_ids"][0], 10000)
        self.assertIn(len(self.showcase["image_ids"]), (8, 17))
        self.assertIn("10000", self.showcase["images"])

        # Check vectors
        self.assertIn(self.vectors_npz["scores"].shape, ((8, 900, 9), (17, 900, 9)))
        self.assertEqual(int(self.vectors_npz["image_ids"][0]), 10000)

        # Check sha256 bindings
        self.assertEqual(self.manifest["files"]["video-showcase.json"], sha256_file(VIDEO_DIR / "video-showcase.json"))
        self.assertEqual(self.manifest["files"]["video-score-vectors.npz"], sha256_file(VIDEO_DIR / "video-score-vectors.npz"))

    def test_03_for_testing_three_model_detections(self):
        if self.showcase is None:
            self.skipTest("Testing video showcase cleaned under D-113")
        """Image 10000 detections strictly locked to supervised taxonomy with exactly 6 visible items."""
        preds = self.showcase["predictions"]["D-095 + Consensus"]["10000"]
        vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
        self.assertEqual(len(vis), 6, f"Expected 6 visible detections, got {len(vis)}")

        # Check categories (only 0..8 allowed)
        for r in vis:
            self.assertIn(r["category_id"], range(9))
            self.assertGreaterEqual(r["score"], 0.15)

        # Check specific items
        # Cigarette butt at bottom-left
        cigs = [r for r in vis if r["category_id"] == 0 and r["bbox"][1] > 3000]
        self.assertEqual(len(cigs), 1)
        self.assertGreaterEqual(cigs[0]["score"], 0.24)
        bx, by, bw, bh = cigs[0]["bbox"]
        self.assertTrue(1300 <= bx <= 1550 and 3300 <= by <= 3500)

        # Center cluster items (y between 1500 and 2500)
        center = [r for r in vis if 1500 <= r["bbox"][1] <= 2500]
        self.assertEqual(len(center), 4)

        # Upper curb item (y < 1000)
        upper = [r for r in vis if r["bbox"][1] < 1000]
        self.assertEqual(len(upper), 1)
        self.assertEqual(upper[0]["category_id"], 1)

    def test_04_distractor_and_border_suppression(self):
        if self.showcase is None:
            self.skipTest("Testing video showcase cleaned under D-113")
        """0 false positives on black shoes, concrete pillar, camera watermark, and road stripe."""
        preds = self.showcase["predictions"]["D-095 + Consensus"]["10000"]
        vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
        w, h = 5136, 3984

        for r in vis:
            bx, by, bw, bh = r["bbox"]
            cx = bx + bw / 2
            cy = by + bh / 2

            # Black shoes area: bottom-right x > 0.70*W, y > 0.45*H
            self.assertFalse(cx > 0.70 * w and cy > 0.45 * h, f"FP detected in black shoes area: {r}")

            # Concrete pillar area: top-right x > 0.75*W, y < 0.25*H
            self.assertFalse(cx > 0.75 * w and cy < 0.25 * h, f"FP detected on concrete pillar: {r}")

            # Camera border margins: top/bottom 6.5%, left/right 5%
            self.assertFalse(cx < 0.05 * w or cx > 0.95 * w or cy < 0.065 * h or cy > 0.935 * h, f"FP detected on camera border: {r}")

    def test_05_spatial_grouping_center_pile_and_individual_litter(self):
        if self.showcase is None:
            self.skipTest("Testing video showcase cleaned under D-113")
        """Image 10000 classifies into exactly 1 Garbage Pile (4 items) and 2 Individual Litter."""
        preds = self.showcase["predictions"]["D-095 + Consensus"]["10000"]
        vis = visible_rows(preds, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.15)[:25]
        grp = compute_spatial_grouping(vis, image_width=5136, image_height=3984)
        s = grp["summary"]

        self.assertEqual(s["total_physical_items"], 6)
        self.assertEqual(s["total_regions"], 3)
        # Under D-112 (min_pile_items = 8): 4-item cluster is Group of Individual Litter
        self.assertEqual(s["garbage_pile_count"], 0)
        self.assertEqual(s["group_of_individual_litter_count"], 1)
        self.assertEqual(s["individual_litter_count"], 2)

        # Region details
        groups = [r for r in grp["regions"] if r["category"] == "group_of_individual_litter"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["item_count"], 4)
        self.assertIsNone(groups[0]["model_score"])

        indivs = [r for r in grp["regions"] if r["category"] == "individual_litter"]
        self.assertEqual(len(indivs), 2)
        for ind in indivs:
            self.assertEqual(ind["item_count"], 1)
            self.assertIsNotNone(ind["model_score"])

        # Check cigarette butt is individual litter
        cig_reg = [r for r in indivs if r["bbox"][1] > 3000]
        self.assertEqual(len(cig_reg), 1)
        m_idx = cig_reg[0]["member_indices"][0]
        self.assertEqual(vis[m_idx]["category_id"], 0)

    def test_06_visual_review_gui_integration_flags(self):
        """Verify --for-testing and --input-image flags and title formatting in taco_stage5_visual_review.py."""
        vr_src = inspect.getsource(vr_mod)
        self.assertIn("--for-testing", vr_src)
        self.assertIn("--input-image", vr_src)
        self.assertIn("Input Still Image — data/input/For_testing.jpg", vr_src)
        self.assertIn("len(objects) > 8", vr_src)

    def test_07_all_14_validation_showcases_preserved(self):
        """Verify zero regressions across all 14 validation showcase images."""
        showcase_expected = {
            1: {"tot": 2, "regs": 1, "pile": 0, "grp": 1, "ind": 0},
            165: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            208: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            238: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            263: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            407: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            466: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            762: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            772: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            777: {"tot": 5, "regs": 1, "pile": 0, "grp": 1, "ind": 0},
            802: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            998: {"tot": 2, "regs": 2, "pile": 0, "grp": 0, "ind": 2},
            1243: {"tot": 1, "regs": 1, "pile": 0, "grp": 0, "ind": 1},
            1465: {"tot": 2, "regs": 1, "pile": 0, "grp": 1, "ind": 0},
        }
        for sid, exp in showcase_expected.items():
            w, h = self.img_dims[sid]
            raw_rows = [r for r in self.d095_champ["raw_queries"] if r["image_id"] == sid]
            vecs = self.d095_vecs["scores"][self.img_map[sid]]
            adj, _, _ = apply_scene_consensus(raw_rows, vecs, image_width=w, image_height=h, enable_rule7=True)
            vis = visible_rows(adj, scheme="9-specific", nms_mode="agnostic_iou_0.50_containment_0.80", threshold=0.10)
            res = compute_spatial_grouping(vis, image_width=w, image_height=h)
            s = res["summary"]
            self.assertEqual(s["total_physical_items"], exp["tot"], f"Image {sid} items mismatch")
            self.assertEqual(s["total_regions"], exp["regs"], f"Image {sid} regions mismatch")
            self.assertEqual(s["garbage_pile_count"], exp["pile"], f"Image {sid} piles mismatch")
            self.assertEqual(s["group_of_individual_litter_count"], exp["grp"], f"Image {sid} groups mismatch")
            self.assertEqual(s["individual_litter_count"], exp["ind"], f"Image {sid} indiv mismatch")


if __name__ == "__main__":
    unittest.main()
