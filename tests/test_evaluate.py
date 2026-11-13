import json
import unittest

from evaluate import outcome, score


def ans(crop="Rice", is_plant=True, **flags):
    return json.dumps({"is_plant": is_plant, "crop": crop, **flags})


def case(i, group, crop=""):
    return {"id": i, "group": group, "crop": crop}


class OutcomeTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(outcome(ans("Maize"))["crop"], "Maize")
        self.assertEqual(outcome(ans("Other"))["kind"], "unsupported")
        self.assertEqual(outcome(ans("Unclear"))["kind"], "unclear")
        self.assertEqual(outcome(ans("Other", is_plant=False))["kind"], "not_plant")
        self.assertEqual(outcome("garbage")["kind"], "invalid")
        self.assertEqual(outcome(ans(blurred=True))["flags"], ["blurred"])


class LenientTests(unittest.TestCase):
    def test_a_fenced_answer_is_invalid_strictly_and_valid_leniently(self):
        fenced = "```json\n" + ans("Rice") + "\n```"
        self.assertEqual(outcome(fenced)["kind"], "invalid")
        self.assertEqual(outcome(fenced, lenient=True)["crop"], "Rice")
        report = score([case("a", "crop", "Rice")], {"a": fenced}, lenient=True)
        self.assertEqual(report["crop_identification"], 1)
        self.assertEqual(score([case("a", "crop", "Rice")], {"a": fenced})["invalid_share"], 1)

    def test_leniency_does_not_rescue_prose_without_an_object(self):
        self.assertEqual(outcome("It is rice.", lenient=True)["kind"], "invalid")


class ScoreTests(unittest.TestCase):
    def test_every_measure_on_a_small_set(self):
        cases = [case("a", "crop", "Rice"), case("b", "crop", "Rice"), case("c", "crop", "Wheat"), case("d", "crop", "Wheat"),
                 case("e", "other-plant"), case("f", "other-plant"), case("g", "not-plant"), case("h", "poor")]
        answers = {"a": ans("Rice"), "b": ans("Maize"), "c": ans("Unclear"), "d": "oops",
                   "e": ans("Other"), "f": ans("Wheat"), "g": ans("Other", is_plant=False), "h": ans("Rice", blurred=True)}
        report = score(cases, answers)
        self.assertEqual(report["crop_identification"], 1 / 4)
        self.assertEqual(report["unclear_share"], 1 / 4)
        self.assertEqual(report["false_stops"], 0)
        self.assertEqual(report["unsupported_plant_stopped"], 1 / 2)
        self.assertEqual(report["non_plant_stopped"], 1)
        self.assertEqual(report["invalid_share"], 1 / 8)
        self.assertEqual(report["quality_raised_on_poor"], 1)
        self.assertEqual(report["per_crop"]["Wheat"], {"unclear": 1, "invalid": 1})
        self.assertEqual(report["confusions"], {"Rice -> Maize": 1})

    def test_a_missing_answer_is_an_error_not_a_zero(self):
        with self.assertRaises(KeyError):
            score([case("a", "crop", "Rice")], {})

    def test_stopping_a_real_crop_photo_is_a_false_stop(self):
        report = score([case("a", "crop", "Rice")], {"a": ans("Other")})
        self.assertEqual((report["false_stops"], report["crop_identification"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
