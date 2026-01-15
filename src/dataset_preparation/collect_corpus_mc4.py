import json
import re
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "raw_hu_sentences.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SENTENCES = 30000


# ------------------------
#  Streaming from mC4 (hu)
# ------------------------

def stream_mc4_hu():
    """
    Stream Hungarian documents from the mC4 'hu' split.
    """
    ds = load_dataset(
        "mc4",
        "hu",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    for ex in ds:
        text = ex.get("text", "")
        if text:
            yield text, "mc4"


# ------------------------
#  Cleaning + Boilerplate removal +sentence split
# ------------------------

BOILERPLATE_PATTERNS = [
    r"cookie-kat használ",
    r"adatvédelmi tájékoztató",
    r"felhasználási feltételek",
    r"minden jog fenntartva",
    r"hírlevél feliratkozás",
    r"oldal tetejére",
]

def looks_like_boilerplate(text: str) -> bool:
    """
    Heuristic boilerplate / navigation / footer detection.
    """
    lower = text.lower()
    tok_len = len(lower.split())
    if tok_len < 5 or tok_len > 60:
        return True

    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, lower):
            return True

    if "http://" in lower or "https://" in lower or "www." in lower:
        return True
    if "@" in lower:
        return True
    if sum(ch in "#|/\\[]" for ch in lower) > 3:
        return True

    return False


def normalize_whitespace(text: str) -> str:
    """
    Normalize Unicode spaces and collapse multiple spaces.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str):
    """
    Simple rule-based sentence splitter (good enough for GEC).
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ------------------------
#  Numeric / format filters
# ------------------------

def digit_ratio_too_high(sent: str, max_ratio: float = 0.25) -> bool:
    """
    Filter out sentences dominated by digits (character level).
    """
    digits = sum(ch.isdigit() for ch in sent)
    if digits == 0:
        return False
    ratio = digits / max(1, len(sent))
    return ratio > max_ratio


def filter_sentence(sent: str) -> bool:
    """
    Return True if the sentence is kept; False if filtered out.
    """
    s = sent.strip()
    if not s:
        return False

    if looks_like_boilerplate(s):
        return False

    if digit_ratio_too_high(s):
        return False

    tokens = s.split()

    # (1) Too many tokens containing digits → likely codes, SKUs, etc.
    digit_tokens = 0
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            digit_tokens += 1

    if tokens and digit_tokens / len(tokens) > 0.05:
        return False

    # (2) Too many uppercase letters → shouty / codes / acronyms
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.6:
            return False

    # (3) Length filter (too short / too long)
    tok_len = len(tokens)
    if tok_len < 5 or tok_len > 40:
        return False

    return True


# ------------------------
#  Main collection pipeline
# ------------------------

def main():
    """
    Stream mC4(hu), split into sentences, filter, and dump JSONL.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out_f = OUT_PATH.open("w", encoding="utf-8")
    n_sent = 0
    sent_id = 0

    for text, source in stream_mc4_hu():
        for sent in split_into_sentences(text):
            if not filter_sentence(sent):
                continue

            rec = {
                "id": sent_id,
                "sentence": sent,
                "source": source,  # "mc4"
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sent_id += 1
            n_sent += 1

            if n_sent % 1000 == 0:
                print(f"Collected {n_sent} sentences...")

            if n_sent >= TARGET_SENTENCES:
                break

        if n_sent >= TARGET_SENTENCES:
            break

    out_f.close()
    print(f"Saved {n_sent} sentences to {OUT_PATH}")


if __name__ == "__main__":
    main()
