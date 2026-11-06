import inspect
import unittest

from scripts.taco_stage5_display_profiles import (
    NMS_CHOICES, THRESHOLD_CHOICES, score_floor, visible_rows,
)


def row(query, category, score, box):
    return {"query_id": query, "category_id": category, "score": score,
            "bbox": box, "valid_clipped_box": True}


class Stage5Image777CalibrationTests(unittest.TestCase):
    def test_calibrated_floor_keeps_hazard_below_macro_floor(self):
        hazard = row(0, 0, .07, [0, 0, 10, 10])
        macro = row(1, 8, .14, [20, 0, 10, 10])
        self.assertEqual(score_floor(hazard, .10, THRESHOLD_CHOICES[1]), .06)
        self.assertEqual(score_floor(macro, .10, THRESHOLD_CHOICES[1]), .15)
        self.assertEqual(score_floor(macro, .20, THRESHOLD_CHOICES[1]), .20)
        self.assertEqual(score_floor(hazard, .20, THRESHOLD_CHOICES[0]), .20)

    def test_cross_class_nms_suppresses_duplicate_material_labels(self):
        plastic = row(1, 4, .40, [0, 0, 100, 100])
        paper = row(2, 8, .35, [1, 1, 98, 98])
        self.assertEqual(len(visible_rows([plastic, paper], "5-material",
                                         "class_aware_iou_0.50", .05)), 2)
        self.assertEqual(len(visible_rows([plastic, paper], "5-material",
                                         "agnostic_iou_0.50", .05)), 1)
        nested = row(3, 8, .30, [10, 10, 10, 10])
        self.assertEqual(len(visible_rows([plastic, nested], "5-material",
                                         "agnostic_iou_0.50_containment_0.80", .05)), 1)

    def test_selector_contract_and_gui_use(self):
        from scripts import taco_stage5_visual_review as viewer
        source = inspect.getsource(viewer.Stage5ViewerGUI)
        self.assertEqual(NMS_CHOICES, (
            "raw", "class_aware_iou_0.50", "class_aware_iou_0.50_containment_0.80",
            "agnostic_iou_0.50", "agnostic_iou_0.50_containment_0.80"))
        for marker in ("nms_var", "threshold_mode_var", "visible_rows", "THRESHOLD_CHOICES"):
            self.assertIn(marker, source)
        self.assertIn("value='agnostic_iou_0.50_containment_0.80'", source)


if __name__ == "__main__":
    unittest.main()
