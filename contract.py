"""The photo-check answer contract, the same shape as the portal's photo check so results are comparable.
An answer that does not parse into it is invalid and counts as a failure."""

import json
from pathlib import Path

import yaml

CROPS = ("Rice", "Maize", "Cotton", "Wheat", "Sugarcane")
CROP_ANSWERS = CROPS + ("Other", "Unclear")
FLAGS = ("blurred", "too_dark", "not_close_up", "several_leaves")
CONTRACT_VERSION = 1


def load_taxonomy(path=Path(__file__).parent / "configs" / "taxonomy.yaml"):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    screening = data["screening_start_set"]
    unknown = set(screening) - set(data["classes"])
    if unknown:
        raise ValueError(f"screening_start_set names classes not in the taxonomy: {sorted(unknown)}")
    return data


def unwrap(text):
    """Lenient reading for baselines: the JSON object inside a code fence or surrounded by prose."""
    if not isinstance(text, str) or "{" not in text or "}" not in text:
        return text
    return text[text.index("{"):text.rindex("}") + 1]


def parse_photo_check(text, lenient=False):
    """Return (answer, None) for a valid answer, or (None, reason). Extra keys, wrong types and bad values are invalid.
    `lenient` first cuts the object out of a code fence or surrounding prose; the default reads the text as given."""
    try:
        data = json.loads(unwrap(text) if lenient else text)
    except (TypeError, ValueError):
        return None, "not_json"
    if not isinstance(data, dict):
        return None, "not_an_object"
    allowed = {"is_plant", "crop", "other_plant", *FLAGS}
    if set(data) - allowed:
        return None, "unexpected_keys"
    if not isinstance(data.get("is_plant"), bool):
        return None, "is_plant_not_bool"
    if data.get("crop") not in CROP_ANSWERS:
        return None, "bad_crop"
    other = data.get("other_plant", "")
    if not isinstance(other, str) or len(other) > 60:
        return None, "bad_other_plant"
    if data["crop"] != "Other" and other:
        return None, "other_plant_without_other"
    if not data["is_plant"] and data["crop"] not in ("Other", "Unclear"):
        return None, "crop_named_for_non_plant"
    flags = {name: data.get(name, False) for name in FLAGS}
    if not all(isinstance(value, bool) for value in flags.values()):
        return None, "flag_not_bool"
    return {"is_plant": data["is_plant"], "crop": data["crop"], "other_plant": other, **flags}, None


def render_photo_check(answer):
    """The canonical training target for an answer: fixed key order, compact JSON."""
    parsed, reason = parse_photo_check(json.dumps(answer))
    if reason:
        raise ValueError(f"Not a valid photo-check answer: {reason}")
    return json.dumps(parsed, separators=(",", ":"))
