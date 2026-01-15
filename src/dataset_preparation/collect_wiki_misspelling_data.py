#!/usr/bin/env python3
# build_dict.py
#
# Extract frequent spelling mistakes from Hungarian Wikipedia
# (wrong -> correct) and save to: data/wiki_misspellings.jsonl
#
# OFFLINE-FRIENDLY: if cached HTML exists (data/wiki_misspellings_raw.html),
# we use that by default instead of downloading.

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "wiki_misspellings.jsonl"
HTML_CACHE = DATA_DIR / "wiki_misspellings_raw.html"

URL = "https://hu.wikipedia.org/wiki/Wikip%C3%A9dia:Helyes%C3%ADr%C3%A1s/Gyakori_el%C3%ADr%C3%A1sok_list%C3%A1ja"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HuGEC-Bot/0.1; +https://example.com/contact)"
}

QUOTE_CHARS = '„”"\'`'

# String patterns where explanations typically start → everything after is dropped
CUTTER_PATTERNS = [
    r"\s+–\s+",  # dash explanation
    r"\s+-\s+",
    r"\s*;\s*",
    r"\s*\|\s*",
    r"\s*(?:Példa|példa)\s*:\s*",
    r"\s*(?:Helyesen|helyesen)\s*:\s*",
    r"\s*(?:Helytelen|helytelen)\s*:\s*",
    r"\s*(?:Megjegyzés|megjegyzés)\s*:\s*",
    r"\s*(?:s még|és még)\s*:\s*",
    # after "lásd még" we drop the rest
    r"\b(?:lásd\s+még|lásd)\b\s*:?\s*",
]

# Triggers indicating that the right side is more like an explanation
BAD_RIGHT_TRIGGERS = re.compile(
    r"\b(vessző|kötőszó|mondat|állítmány|ha\s+összehasonlítás|akkor\s+és\s+csak\s+akkor|példa)\b",
    re.IGNORECASE,
)

# Remove simple (...) parenthetical segments
PAREN_RE = re.compile(r"\([^)]*\)")


def norm_space(s: str) -> str:
    """Normalize whitespace and convert NBSP to regular space."""
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def strip_outer_quotes(s: str) -> str:
    """Remove only outermost quote characters, keep inner quotes."""
    s = s.strip()
    while len(s) >= 2 and s[0] in QUOTE_CHARS and s[-1] in QUOTE_CHARS:
        s = s[1:-1].strip()
    return s


def remove_parentheses(s: str) -> str:
    """Strip simple (...) segments and re-normalize whitespace."""
    s = norm_space(s)
    s = PAREN_RE.sub("", s)
    return norm_space(s)


def cut_explanations(s: str) -> str:
    """Drop explanatory tail after typical separators (dash, semicolon, etc.)."""
    s = norm_space(s)
    for pat in CUTTER_PATTERNS:
        s = re.split(pat, s, maxsplit=1)[0]
        s = norm_space(s)
    return s


def looks_like_sentence(s: str) -> bool:
    """Heuristic: if it looks like a full sentence, treat as explanation."""
    s = norm_space(s)
    if re.search(r"[.!?]$", s):
        return True
    # long multi-word sequences are more likely commentary than a single term
    if len(s.split()) >= 6:
        return True
    return False


def is_bad_pair_candidate(wrong: str, right: str) -> bool:
    """
    Decide whether this is likely not a simple dictionary-like pair
    but a rule explanation / partial phrase.
    """
    wrong_n = norm_space(strip_outer_quotes(wrong))
    right_n = norm_space(strip_outer_quotes(right))

    if len(wrong_n) <= 3:
        return True

    if BAD_RIGHT_TRIGGERS.search(right_n):
        return True
    if looks_like_sentence(right_n):
        return True

    return False


def extract_pairs_from_li_text(text: str):
    """
    Extract (wrong, right) spelling pairs from a single <li> item text.

    Handles:
      - cutting explanations on the right side
      - removing parentheses
      - splitting multiple wrong variants (vagy/avagy, commas)
      - mapping multiple right variants when possible.
    """
    text = norm_space(text)
    if "→" not in text:
        return []

    left, right = text.split("→", 1)
    left = norm_space(left)
    right = norm_space(right)

    # 1) remove parentheses on both sides
    left = remove_parentheses(left)
    right = remove_parentheses(right)

    # 2) drop explanatory tail on the right
    right = cut_explanations(right)

    # 3) cut at "avagy" on the right (mostly explanatory)
    right = re.split(r"\bavagy\b", right, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    right = norm_space(right)

    # 4) strip outer quotes
    left = strip_outer_quotes(left)
    right = strip_outer_quotes(right)

    # 5) wrong forms: split by comma / vagy / avagy
    wrong_parts = [
        p
        for p in re.split(
            r"\s*(?:,|\bavagy\b|\bvagy\b)\s*", left, flags=re.IGNORECASE
        )
        if p.strip()
    ]
    wrong_forms = [
        remove_parentheses(strip_outer_quotes(norm_space(w.strip(" ~"))))
        for w in wrong_parts
    ]

    # 6) right forms: split by comma
    right_forms = [
        remove_parentheses(strip_outer_quotes(norm_space(r.strip(" ~"))))
        for r in right.split(",")
        if r.strip()
    ]

    pairs = []
    if not wrong_forms or not right_forms:
        return pairs

    if len(right_forms) == 1:
        correct = right_forms[0]
        for w in wrong_forms:
            if w and correct and not is_bad_pair_candidate(w, correct):
                pairs.append((w, correct))
    else:
        for w, c in zip(wrong_forms, right_forms):
            if w and c and not is_bad_pair_candidate(w, c):
                pairs.append((w, c))

    return pairs


def download_html() -> str:
    """
    Download the Wikipedia page and write it to cache.
    If download fails and no cache exists, raise an error.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading Wikipedia page...\n  {URL}")
        resp = requests.get(URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        html = resp.text
        HTML_CACHE.write_text(html, encoding="utf-8")
        print(f"Downloaded OK, cached to {HTML_CACHE}")
        return html
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        if HTML_CACHE.exists():
            print(f"Using cached HTML from {HTML_CACHE}")
            return HTML_CACHE.read_text(encoding="utf-8")
        raise RuntimeError(
            "Could not download the page and no cache is available. "
            f"Please save the HTML manually to: {HTML_CACHE}"
        )


def load_html_prefer_cache() -> str:
    """
    Offline-friendly loader:
      - if cached HTML exists, use it and do not download
      - otherwise fall back to live download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HTML_CACHE.exists():
        print(f"Using cached HTML from {HTML_CACHE} (offline mode)")
        return HTML_CACHE.read_text(encoding="utf-8")
    return download_html()


def main():
    html = load_html_prefer_cache()
    soup = BeautifulSoup(html, "html.parser")

    # Iterate over all <li> and only keep those containing "→".
    entries = []
    seen_pairs = set()

    li_count = 0
    arrow_li_count = 0
    rejected_estimate = 0

    for li in soup.find_all("li"):
        li_count += 1
        text = li.get_text(" ", strip=True)
        if "→" not in text:
            continue
        arrow_li_count += 1

        pairs = extract_pairs_from_li_text(text)
        if not pairs:
            rejected_estimate += 1

        for wrong, right in pairs:
            key = (wrong, right)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            entries.append({"wrong": wrong, "right": right})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in entries:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Scanned <li>: {li_count}")
    print(f"<li> containing '→': {arrow_li_count}")
    print(f"Saved {len(entries)} unique (wrong->right) pairs to: {OUT_PATH}")
    print(
        f"Arrow-<li> with 0 extracted pairs (explanations / noise approx.): "
        f"{rejected_estimate}"
    )
    print("Sample pairs:")
    for ex in entries[:10]:
        print(" ", ex)


if __name__ == "__main__":
    main()
