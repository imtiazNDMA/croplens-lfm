import json
import unittest

from contract import load_taxonomy, parse_photo_check, render_photo_check

GOOD = {"is_plant": True, "crop": "Rice", "other_plant": "", "blurred": False, "too_dark": False,
        "not_close_up": False, "several_leaves": False}


class ContractTests(unittest.TestCase):
    def test_taxonomy_is_frozen_with_the_audited_start_set(self):
        data = load_taxonomy()
        self.assertEqual(len(data["classes"]), 39)
        self.assertEqual(len(data["screening_start_set"]), 18)
        self.assertEqual(data["taxonomy_version"], 1)
        self.assertEqual({data["classes"][c]["crop"] for c in data["screening_start_set"]},
                         {"Cotton", "Maize", "Rice", "Sugarcane", "Wheat"})

    def test_valid_answer_round_trips_and_defaults_missing_fields(self):
        answer, reason = parse_photo_check(json.dumps({"is_plant": True, "crop": "Wheat"}))
        self.assertIsNone(reason)
        self.assertEqual(answer["crop"], "Wheat")
        self.assertFalse(answer["blurred"])
        self.assertEqual(parse_photo_check(render_photo_check(GOOD))[0], GOOD)

    def test_invalid_answers_name_their_reason(self):
        cases = {
            "not json": "not_json", "[1]": "not_an_object",
            json.dumps({**GOOD, "extra": 1}): "unexpected_keys",
            json.dumps({**GOOD, "is_plant": "yes"}): "is_plant_not_bool",
            json.dumps({**GOOD, "crop": "Barley"}): "bad_crop",
            json.dumps({**GOOD, "other_plant": "x" * 61, "crop": "Other"}): "bad_other_plant",
            json.dumps({**GOOD, "other_plant": "tomato"}): "other_plant_without_other",
            json.dumps({**GOOD, "is_plant": False}): "crop_named_for_non_plant",
            json.dumps({**GOOD, "blurred": "no"}): "flag_not_bool",
        }
        for text, expected in cases.items():
            self.assertEqual(parse_photo_check(text), (None, expected), text)
        self.assertEqual(parse_photo_check(None), (None, "not_json"))

    def test_non_plant_may_say_other_or_unclear(self):
        for crop in ("Other", "Unclear"):
            self.assertIsNone(parse_photo_check(json.dumps({**GOOD, "is_plant": False, "crop": crop}))[1])

    def test_render_refuses_an_invalid_answer(self):
        with self.assertRaises(ValueError):
            render_photo_check({**GOOD, "crop": "Barley"})


if __name__ == "__main__":
    unittest.main()
