from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset
from dataset_preparation.paralel_dep_transfer import generate_paralel_syntax_error

# ---- Paths / constants ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_PATH = DATA_DIR / "gec_pairs_paralel_de_hu.jsonl"

SRC_LANG = "de"
TGT_LANG = "hu"

# maximum number of paralel GEC pairs to generate
TARGET_N = 1000

# maximum number of raw de–hu pairs to process (after filtering)
MAX_PAIRS = 2000


# ---- Simple cleaning / filtering (mirrors old build script) -----------------

BAD_SUBSTRINGS = ["<html", "<body", "</", "http://", "https://"]


def looks_boilerplate(text: str) -> bool:
    """Simple boilerplate / HTML-ish snippet detection."""
    lower = text.lower()
    return any(bad in lower for bad in BAD_SUBSTRINGS)


def too_short(text: str, min_tokens: int = 3) -> bool:
    """Discard sentences with fewer than min_tokens whitespace tokens."""
    return len(text.strip().split()) < min_tokens


def normalize_space(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    return " ".join(text.strip().split())


def is_good_pair(src: str, tgt: str) -> bool:
    """Basic well-formedness checks for a de–hu parallel sentence pair."""
    if not src or not tgt:
        return False
    if too_short(src) or too_short(tgt):
        return False
    if looks_boilerplate(src) or looks_boilerplate(tgt):
        return False
    return True


# ---- Capitalization fix (shared with other script) --------------------------

def fix_sentence_capitalization(orig_correct: str, incorrect: str) -> str:
    """
    Heuristic capitalization handling after paralel word-order changes.
    """
    orig_tokens = orig_correct.strip().split()
    inc_tokens = incorrect.strip().split()

    if not orig_tokens or not inc_tokens:
        return incorrect

    orig_first = orig_tokens[0]

    # 1) ensure the new first token starts with an uppercase letter
    t0 = inc_tokens[0]
    if t0 and t0[0].isalpha():
        inc_tokens[0] = t0[0].upper() + t0[1:]

    # 2) original first token, if moved inside, usually becomes lowercase
    if orig_first and orig_first[0].isupper():
        for i in range(1, len(inc_tokens)):
            if inc_tokens[i] == orig_first:
                tok = inc_tokens[i]
                if tok.isupper() or any(ch.isdigit() for ch in tok) or "." in tok:
                    continue
                inc_tokens[i] = tok[0].lower() + tok[1:]

    return " ".join(inc_tokens)


# ---- Main -------------------------------------------------------------------

def main():
    random.seed(42)

    print("Loading OPUS Books de-hu from HuggingFace...")
    ds = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-hu",
        split="train",
        trust_remote_code=True,
    )

    # 1) collect filtered de–hu pairs, deduplicate, then shuffle
    pairs = []
    dedup_set = set()
    n_seen = 0

    for ex in ds:
        n_seen += 1
        trans = ex["translation"]
        src = normalize_space(trans[SRC_LANG])
        tgt = normalize_space(trans[TGT_LANG])

        if not is_good_pair(src, tgt):
            continue

        key = (src, tgt)
        if key in dedup_set:
            continue
        dedup_set.add(key)

        pairs.append(
            {
                "id": len(pairs),
                "src_lang": SRC_LANG,
                "src": src,
                "tgt": tgt,
                "source": "Helsinki-NLP/opus_books",
            }
        )

        if len(pairs) >= MAX_PAIRS:
            break

    random.shuffle(pairs)
    print(f"Collected {len(pairs)} clean de-hu pairs (seen {n_seen})")

    # 2) generate paralel word-order errors and write GEC pairs
    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for ex in pairs:
            src_lang = ex["src_lang"]  # "de"
            src = ex["src"]
            hu = ex["tgt"]

            result = generate_paralel_syntax_error(
                de_sentence=src,
                hu_sentence=hu,
            )
            if result is None:
                continue

            incorrect, error_types = result
            incorrect = fix_sentence_capitalization(hu, incorrect)
            if incorrect.strip() == hu.strip():
                continue

            rec = {
                "id": ex["id"],       # ID of the original parallel pair
                "correct": hu,
                "incorrect": incorrect,
                "meta": {
                    "strategy": "paralel_dep_transfer",
                    "error_types": error_types,
                    "src_lang": src_lang,
                    "src": src,
                    "source": ex["source"],
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 100 == 0:
                print(f"Generated {n_written} paralel GEC pairs...")

            if n_written >= TARGET_N:
                break

    print(f"Done. Saved {n_written} paralel GEC pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
