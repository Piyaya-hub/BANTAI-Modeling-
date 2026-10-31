import unittest

from scripts.taco_stage5_organic_audit import disposition, mapping_for, resized_box


class OrganicAuditTests(unittest.TestCase):
    categories = {1: "Plastic film", 2: "Food waste", 3: "Food Can",
                  4: "Disposable food container", 5: "Plastic bottle cap",
                  6: "Unlabeled litter", 7: "Normal paper"}
    image = {"status": "readable", "issues": [], "actual_width": 2448,
             "actual_height": 3264, "exif_orientation": 1}

    def ann(self, ident, category, box=(10, 20, 5, 6)):
        return {"id": ident, "category_id": category, "bbox": list(box),
                "area": box[2] * box[3], "iscrowd": 0, "ignore": 0}

    def test_food_waste_is_class_nine_and_tiered_sibling_rules_are_preserved(self):
        self.assertEqual(mapping_for(self.categories, "tier1")["Food waste"], 9)
        self.assertNotIn("Food Can", mapping_for(self.categories, "tier1"))
        self.assertEqual(mapping_for(self.categories, "tier2")["Food Can"], 5)
        self.assertEqual(mapping_for(self.categories, "tier2")["Disposable food container"], 6)

    def test_clean_food_annotation_survives_and_sibling_unlocks_only_tier2(self):
        food = self.ann(10, 2)
        self.assertTrue(disposition(self.image, [food], self.categories, "tier1")["eligible_before_duplicate_rules"])
        with_food_can = [food, self.ann(11, 3)]
        self.assertEqual(disposition(self.image, with_food_can, self.categories, "tier1")["primary_reason"],
                         "excluded_annotation")
        self.assertTrue(disposition(self.image, with_food_can, self.categories, "tier2")["eligible_before_duplicate_rules"])

    def test_unlabeled_blocks_even_when_tier3_omits_benign_accessory(self):
        rows = [self.ann(10, 2), self.ann(11, 5), self.ann(12, 6)]
        result = disposition(self.image, rows, self.categories, "tier3")
        self.assertEqual(result["primary_reason"], "unlabeled_annotation")
        self.assertEqual(result["omitted_accessory_annotation_ids"], [11])
        self.assertFalse(result["eligible_before_duplicate_rules"])

    def test_resized_box_uses_short_edge_and_preserves_aspect(self):
        result = resized_box([100, 200, 50, 40], 2448, 3264)
        self.assertAlmostEqual(result["scale"], 384 / 2448)
        self.assertAlmostEqual(result["width_px"], 50 * 384 / 2448)
        self.assertAlmostEqual(result["height_px"], 40 * 384 / 2448)


if __name__ == "__main__":
    unittest.main()
