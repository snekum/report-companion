"""Shared pieces for baseline, generation, training and evaluation.

Pure standard library so the deterministic baseline and the scorer run
without the training environment.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = open(os.path.join(ROOT, "shared", "extraction_prompt.txt"), encoding="utf-8").read().strip()
DATA = os.path.join(ROOT, "data", "processed")
SECTIONS = ("labs", "medications", "follow_up")


def messages_for(text: str) -> list[dict]:
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]


def target_json(target: dict) -> str:
    """Canonical serialisation used as the training completion."""
    return json.dumps(target, ensure_ascii=False, indent=None, separators=(", ", ": "))


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ parsing
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_output(raw: str):
    """Returns (obj or None, error or None). Tolerates a think block and code fences."""
    s = THINK_RE.sub("", raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    start = s.find("{")
    if start < 0:
        return None, "no JSON object found"
    depth, end, in_str, esc = 0, -1, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None, "unterminated JSON (output probably hit the token limit)"
    try:
        return json.loads(s[start:end + 1]), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg}"


def _str_or_null(v):
    return v is None or isinstance(v, str)


def schema_errors(obj) -> list[str]:
    """Hand-written check mirroring shared/extraction.schema.json."""
    errs = []
    if not isinstance(obj, dict):
        return ["top level is not an object"]
    keys = {"document_date", "labs", "medications", "follow_up"}
    if set(obj) != keys:
        errs.append(f"keys {sorted(obj)} != {sorted(keys)}")
    dd = obj.get("document_date")
    if dd is not None:
        if not (isinstance(dd, dict) and set(dd) == {"value", "evidence"} and all(isinstance(dd[k], str) and dd[k] for k in dd)):
            errs.append("document_date malformed")
    spec = {
        "labs": ({"test", "value", "unit", "reference_range", "date", "evidence"}, {"test", "value", "evidence"}),
        "medications": ({"name", "dose", "frequency", "status", "evidence"}, {"name", "evidence"}),
        "follow_up": ({"instruction", "evidence"}, {"instruction", "evidence"}),
    }
    for sec, (fields, required_str) in spec.items():
        arr = obj.get(sec)
        if not isinstance(arr, list):
            errs.append(f"{sec} is not a list")
            continue
        for i, it in enumerate(arr):
            if not isinstance(it, dict) or set(it) != fields:
                errs.append(f"{sec}[{i}] has wrong fields")
                continue
            for k in fields:
                if k in required_str:
                    if not (isinstance(it[k], str) and it[k].strip()):
                        errs.append(f"{sec}[{i}].{k} must be a non-empty string")
                elif k == "status":
                    if it[k] not in ("current", "not_current", "unclear"):
                        errs.append(f"{sec}[{i}].status invalid")
                elif not _str_or_null(it[k]):
                    errs.append(f"{sec}[{i}].{k} must be string or null")
    return errs


def coerce(obj) -> dict:
    """Best-effort view of a schema-invalid object, so partially valid output can still be scored.
    Items that are not objects are dropped; missing fields become None."""
    out = {"document_date": None, "labs": [], "medications": [], "follow_up": []}
    if not isinstance(obj, dict):
        return out
    dd = obj.get("document_date")
    if isinstance(dd, dict) and isinstance(dd.get("value"), str):
        out["document_date"] = {"value": dd["value"], "evidence": dd.get("evidence") if isinstance(dd.get("evidence"), str) else None}
    fields = {
        "labs": ["test", "value", "unit", "reference_range", "date", "evidence"],
        "medications": ["name", "dose", "frequency", "status", "evidence"],
        "follow_up": ["instruction", "evidence"],
    }
    for sec, fs in fields.items():
        for it in obj.get(sec) or []:
            if isinstance(it, dict):
                out[sec].append({f: (it.get(f) if isinstance(it.get(f), str) else None) for f in fs})
    return out


# ------------------------------------------------------------------ support checks
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_key(s: str | None) -> str:
    return norm_ws(s).lower() if s else ""


def evidence_found(source: str, evidence: str | None) -> bool:
    return bool(evidence) and norm_ws(evidence) in norm_ws(source)


def token_in(haystack: str, needle: str, case=True) -> bool:
    """needle occurs in haystack and is not part of a longer number."""
    h, n = norm_ws(haystack), norm_ws(needle)
    if not case:
        h, n = h.lower(), n.lower()
    if not n:
        return False
    pat = re.escape(n)
    if n[0].isdigit() or n[0] in "<>.":
        pat = r"(?<![\d.])" + pat
    if n[-1].isdigit():
        pat = pat + r"(?![\d])"
    return re.search(pat, h) is not None


ITEM_FIELDS = {
    "labs": ("test", ["value", "unit", "reference_range", "date"]),
    "medications": ("name", ["dose", "frequency"]),
    "follow_up": ("instruction", []),
}


def item_support(source: str, sec: str, item: dict) -> dict:
    """Code-side verification of one extracted item.

    evidence_ok:   the quote occurs in the document (whitespace-normalised)
    supported:     every printed field (name/value/unit/range/date/dose/frequency)
                   occurs in the document
    in_evidence:   every printed field occurs inside the quote itself
    """
    key, others = ITEM_FIELDS[sec]
    ev = item.get("evidence")
    ev_ok = evidence_found(source, ev)
    fields = [key] + others
    missing_doc, missing_ev = [], []
    for f in fields:
        v = item.get(f)
        if v is None:
            continue
        case = f not in ("test", "name", "instruction")
        if not token_in(source, v, case=case):
            missing_doc.append(f)
        if not (ev and token_in(ev, v, case=case)):
            missing_ev.append(f)
    return {"evidence_ok": ev_ok, "supported": not missing_doc, "in_evidence": ev_ok and not missing_ev,
            "unsupported_fields": missing_doc, "fields_outside_evidence": missing_ev}


# ------------------------------------------------------------------ matching
def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def names_match(a: str | None, b: str | None) -> bool:
    a, b = norm_key(a), norm_key(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def lab_match(p, g) -> bool:
    return names_match(p["test"], g["test"]) and norm_ws(p["value"]) == norm_ws(g["value"]) and (
        g["date"] is None or norm_key(p["date"]) == norm_key(g["date"]))


def med_match(p, g) -> bool:
    return names_match(p["name"], g["name"])


def fu_match(p, g) -> bool:
    return jaccard(p["instruction"], g["instruction"]) >= 0.6 or (
        norm_key(p["instruction"]) and (norm_key(p["instruction"]) in norm_key(g["instruction"]) or norm_key(g["instruction"]) in norm_key(p["instruction"])) and jaccard(p["instruction"], g["instruction"]) >= 0.4)


MATCHERS = {"labs": lab_match, "medications": med_match, "follow_up": fu_match}


def greedy_pairs(pred: list, gold: list, match) -> list[tuple[int, int]]:
    used, pairs = set(), []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j not in used and match(p, g):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs
