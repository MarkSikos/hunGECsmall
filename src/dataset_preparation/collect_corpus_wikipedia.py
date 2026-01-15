from datasets import load_dataset
from pathlib import Path
import re
import json
import random

random.seed(42)

# ---------- Paths (aligned with the other corpus script) ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "clean_hu_sentences.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Target size + simple sentence length constraints
SENT_MIN_TOKENS = 5
SENT_MAX_TOKENS = 30
TARGET_N_SENT = 8000

# ---------- Basic sentence splitting + filters ----------

def simple_tokenize(text: str):
    """Very simple whitespace-based tokenizer."""
    return [t for t in re.split(r"\s+", text.strip()) if t]


CAPITAL_START_RE = re.compile(r'^[A-ZÁÉÍÓÖŐÚÜŰ0-9]')


def normalize_sentence(s: str) -> str:
    """
    Normalize a raw sentence candidate:
      - split on newlines and pick the best sub-span
      - strip leading bullets / dashes / quotes
      - normalize whitespace.
    """
    parts = re.split(r"\n+", s)
    candidates = []

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # leading dash / bullet
        p = re.sub(r'^[\-\–•\*·,;:]+', "", p).lstrip()
        # leading quotes / brackets
        p = re.sub(r'^[„"\'\(\[]+', "", p).lstrip()
        if not p:
            continue

        candidates.append(p)

    if not candidates:
        return ""

    # choose the "best" piece:
    #   - prefer capitalized start
    #   - then prefer more tokens
    best = None
    best_score = -1
    for c in candidates:
        toks = simple_tokenize(c)
        if not toks:
            continue
        score = len(toks)
        if CAPITAL_START_RE.match(c):
            score += 100
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return ""

    best = re.sub(r"\s+", " ", best).strip()
    return best


def split_into_sentences(text: str):
    """
    Naive Hungarian-friendly sentence splitter on . ! ?,
    followed by basic normalization and length filtering.
    """
    raw_sents = re.split(r"(?<=[\.!\?])\s+", text)
    sents = []
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue

        s = normalize_sentence(s)
        if not s:
            continue

        if not CAPITAL_START_RE.match(s):
            continue

        toks = simple_tokenize(s)
        if SENT_MIN_TOKENS <= len(toks) <= SENT_MAX_TOKENS:
            sents.append(s)
    return sents


# ---------- Simple noise / garbage filter ----------

hungarian_chars = "A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű"
bad_pattern = re.compile(r"https?://|www\.|[@#]")


def looks_ok(sentence: str) -> bool:
    """
    Filter out obviously bad lines: parentheses, URLs, too many odd chars, etc.
    """
    # reject if it contains hyphen or parentheses
    if "-" in sentence or "–" in sentence or "(" in sentence or ")" in sentence:
        return False

    if bad_pattern.search(sentence):
        return False

    # require that at least ~90% of chars are letters/digits/simple punctuation
    cleaned = re.sub(rf"[^{hungarian_chars}\s\.,;:\-!?]", "", sentence)
    ratio = len(cleaned) / max(len(sentence), 1)
    if ratio < 0.9:
        return False

    return True


# ---------- Load Hungarian Wikipedia (streaming) ----------

def load_hu_stream():
    """
    Stream Hungarian Wikipedia text from the Wikimedia dump.
    """
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.hu",          # Hungarian Wikipedia dump
        split="train",
        streaming=True,
    )
    return ds


# ---------- Continue IDs from existing corpus file ----------

def get_next_id() -> int:
    """
    Scan existing corpus file and return the next free integer id.
    If file does not exist or is empty, start from 0.
    """
    if not OUT_PATH.exists():
        return 0

    last_id = -1
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in ex:
                try:
                    last_id = max(last_id, int(ex["id"]))
                except (ValueError, TypeError):
                    continue

    return last_id + 1


# ---------- MAIN ----------

def main():
    """
    Append Hungarian Wikipedia sentences to the existing corpus file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_id = get_next_id()
    print(f"Starting WIKI sentences from id={start_id}")

    dataset_stream = load_hu_stream()

    collected = 0
    sent_id = start_id

    # open in append mode to keep the existing corpus
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for row in dataset_stream:
            text = row["text"]
            for sent in split_into_sentences(text):
                if not looks_ok(sent):
                    continue

                rec = {
                    "id": sent_id,
                    "sentence": sent,
                    "source": "wiki",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                sent_id += 1
                collected += 1

                if collected % 1000 == 0:
                    print(f"Collected {collected} WIKI sentences... (last id={sent_id-1})")

                if collected >= TARGET_N_SENT:
                    break

            if collected >= TARGET_N_SENT:
                break

    print(f"Appended {collected} WIKI sentences to {OUT_PATH}")
    print(f"Last id used: {sent_id-1}")


if __name__ == "__main__":
    main()
