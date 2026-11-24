"""Run with: python -I -m pytest model/test_common.py   (or python model/test_common.py)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, item_support, parse_output, schema_errors  # noqa: E402

CASES = json.load(open(os.path.join(ROOT, "shared", "validation_cases.json"), encoding="utf-8"))


def test_shared_cases():
    for c in CASES["cases"]:
        s = item_support(CASES["source"], c["section"], c["item"])
        assert s["evidence_ok"] == c["evidence_found"], c["name"]
        assert s["in_evidence"] == c["verified"], c["name"]


def test_parse_strips_think_and_fences():
    obj, err = parse_output('<think>\n\n</think>\n\n```json\n{"document_date": null, "labs": [], "medications": [], "follow_up": []}\n```')
    assert err is None and schema_errors(obj) == []


def test_truncated_output():
    obj, err = parse_output('{"document_date": null, "labs": [{"test": "Glu')
    assert obj is None and "token limit" in err


def test_schema_rejects_bare_strings():
    obj, _ = parse_output('{"document_date": null, "labs": [], "medications": [], "follow_up": ["Repeat in 3 months"]}')
    assert schema_errors(obj)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
