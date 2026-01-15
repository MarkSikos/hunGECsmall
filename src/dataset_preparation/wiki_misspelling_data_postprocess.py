from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from utilities.llm_client import call_llm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IN_PATH = DATA_DIR / "wiki_misspellings.jsonl"
OUT_PATH = DATA_DIR / "wiki_misspellings.cleaned.jsonl"
REJ_PATH = DATA_DIR / "wiki_misspellings.rejected.jsonl"

NBSP = "\xa0"
PAREN_RE = re.compile(r"\([^)]*\)")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def norm_space(s: str) -> str:
    """Normalize whitespace and replace NBSP with regular spaces."""
    return re.sub(r"\s+", " ", s.replace(NBSP, " ")).strip()


def preclean(wrong: str, right: str) -> Tuple[str, str]:
    """
    Lightweight heuristic cleanup before sending to the LLM:
    - remove parentheses
    - trim simple explanations from the right side
    """
    wrong = PAREN_RE.sub("", norm_space(wrong))
    right = PAREN_RE.sub("", norm_space(right))

    # cut off simple explanation tails on the right side
    right = re.split(r"\s*;\s*", right, maxsplit=1)[0]
    right = re.split(
        r"\b(?:lásd\s+még|lásd)\b\s*:?\s*",
        right,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    right = re.sub(
        r"^(?:helyesen|helyes|helytelen)\s*:\s*",
        "",
        right,
        flags=re.IGNORECASE,
    )

    return norm_space(wrong), norm_space(right)


SYSTEM = """
Te egy magyar helyesírási hibaszótár tisztító modul vagy.
KIZÁRÓLAG JSON-t adj vissza:
{"keep": true/false, "wrong": "...", "right": "..."}
- keep=true csak ha konkrét hibás->helyes alakpár.
- Ha magyarázat/szabály/példa/törmelék: keep=false.
- wrong/right legyen rövid és konkrét (általában 1–4 szó).
""".strip()


def parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to recover a JSON object from the LLM response:
    - direct parse
    - fallback to first {...} span via regex
    """
    t = text.strip()
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    m = JSON_OBJ_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean_one(wrong: str, right: str) -> Dict[str, Any]:
    """
    Call the LLM once for a (wrong, right) candidate and
    return a dict with resolved keep / wrong / right fields.
    """
    user = f"wrong: {wrong}\nright: {right}\n"
    resp = call_llm(
        system=SYSTEM,
        user=user,
        max_tokens=96,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )
    d = parse_json_obj(resp) or {}
    keep = bool(d.get("keep", False))
    w = norm_space(str(d.get("wrong", wrong)))
    r = norm_space(str(d.get("right", right)))
    if not w or not r:
        keep = False
    return {"keep": keep, "wrong": w, "right": r}


def main() -> None:
    """
    Load raw pairs, pre-clean, LLM-filter them, deduplicate, and
    write cleaned and rejected pairs to separate JSONL files.
    """
    cleaned = []
    rejected = []
    seen = set()

    # One-off header line removal on the raw file (kept as-is).
    from pathlib import Path

    p = Path("data/wiki_misspellings.jsonl")
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            wrong0 = str(row.get("wrong", ""))
            right0 = str(row.get("right", ""))

            wrong, right = preclean(wrong0, right0)
            if not wrong or not right:
                rejected.append({"wrong": wrong0, "right": right0})
                continue

            res = clean_one(wrong, right)
            if res["keep"]:
                key = (res["wrong"], res["right"])
                if key not in seen:
                    seen.add(key)
                    cleaned.append({"wrong": res["wrong"], "right": res["right"]})
            else:
                rejected.append({"wrong": wrong0, "right": right0})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with REJ_PATH.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept={len(cleaned)} -> {OUT_PATH}")
    print(f"rejected={len(rejected)} -> {REJ_PATH}")


if __name__ == "__main__":
    main()
