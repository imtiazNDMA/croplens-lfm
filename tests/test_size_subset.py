import unittest

from size_subset import ambiguous_sizes, subset_ids


class SizeSubsetTests(unittest.TestCase):
    def test_a_size_is_ambiguous_only_when_several_crops_use_it(self):
        crop_by_size = {(256, 256): ["Maize", "Maize"], (300, 300): ["Rice", "Maize"], (10, 10): ["Wheat"]}
        self.assertEqual(ambiguous_sizes(crop_by_size), {(300, 300)})

    def test_subset_keeps_only_crop_photos_at_ambiguous_sizes(self):
        crop_by_size = {(256, 256): ["Maize"], (300, 300): ["Rice", "Maize"]}
        sizes = {"a.jpg": (256, 256), "b.jpg": (300, 300), "c.jpg": (300, 300)}
        cases = [{"id": "A", "group": "crop", "path": "a.jpg"}, {"id": "B", "group": "crop", "path": "b.jpg"},
                 {"id": "C", "group": "poor", "path": "c.jpg"}]
        self.assertEqual(subset_ids(cases, sizes, crop_by_size), ["B"])


if __name__ == "__main__":
    unittest.main()
