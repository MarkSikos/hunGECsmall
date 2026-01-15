from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from utilities.config import DATA_DIR
from grammatical_error_scripts.typo_error import add_char_noise
from prompt_engineering.answer_engineering import score_noisy_candidate

# Path to cross-lingual GEC pairs (created by the OPUS-based script)
paralel_PATH = DATA_DIR / "gec_pairs_paralel_de_hu.jsonl"


# ---------------------------------------------------------------------------
# Load pre-generated paralel pairs
# ---------------------------------------------------------------------------

def load_PARALEL_PAIRS(max_pairs: int = 50_000) -> List[Dict[str, Any]]:
    """
    Load pre-generated paralel GEC pairs from disk.

    Each line is expected to be a JSON object with at least:
      - "correct": str
      - "incorrect": str
    """
    pairs: List[Dict[str, Any]] = []
    if not paralel_PATH.exists():
        return pairs

    with paralel_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "correct" not in ex or "incorrect" not in ex:
                continue

            pairs.append(ex)
            if len(pairs) >= max_pairs:
                break

    return pairs


# Global cache – imported and reused by generators
PARALEL_PAIRS: List[Dict[str, Any]] = load_PARALEL_PAIRS()


# ---------------------------------------------------------------------------
# Optional post char-level noise for paralel variants
# ---------------------------------------------------------------------------

def maybe_add_post_char_noise(
    orig: str,
    rec: Dict[str, Any],
    max_change_ratio: float,
    max_token_ratio: float = 0.15,
    p_apply: float = 0.3,
) -> Dict[str, Any]:
    """
    Optionally add extra character-level noise to an already generated pair.

    - With probability `p_apply` we noised `rec['incorrect']`.
    - If the new candidate scores badly, we keep the original `rec`.
    """
    if random.random() >= p_apply:
        return rec

    base_incorrect = rec["incorrect"]
    noised = add_char_noise(base_incorrect, max_token_ratio=max_token_ratio)

    if not noised or noised == base_incorrect:
        return rec

    score, sim, len_ratio, change_ratio = score_noisy_candidate(
        orig, noised, max_change_ratio=max_change_ratio
    )

    # Very bad candidates are rejected.
    if score <= -1e8:
        return rec

    new_meta = dict(rec["meta"])
    new_meta.update(
        {
            "score": float(score),
            "sim": float(sim),
            "len_ratio": float(len_ratio),
            "change_ratio": float(change_ratio),
            "post_char_noise": True,
            "post_char_noise_max_token_ratio": max_token_ratio,
        }
    )

    new_rec = dict(rec)
    new_rec["incorrect"] = noised
    new_rec["meta"] = new_meta
    return new_rec


# ---------------------------------------------------------------------------
# Public helper to sample one paralel-based GEC example
# ---------------------------------------------------------------------------

def make_paralel_variant() -> Optional[Dict[str, Any]]:
    """
    Pick a pre-generated de→hu paralel pair and adapt it into our GEC format.

    Optionally adds a small amount of post character-level noise.
    """
    if not PARALEL_PAIRS:
        return None

    ex = random.choice(PARALEL_PAIRS)

    orig_correct = ex["correct"].strip()
    incorrect = ex["incorrect"].strip()
    if not orig_correct or len(orig_correct.split()) < 5:
        return None

    base_meta = ex.get("meta", {})
    base_meta["strategy"] = base_meta.get("strategy", "paralel_de_transfer")

    rec = {
        "correct": orig_correct,
        "incorrect": incorrect,
        "meta": base_meta,
    }

    # Optional extra character-level noise on top of the paralel variant
    rec = maybe_add_post_char_noise(
        orig=orig_correct,
        rec=rec,
        max_change_ratio=0.6,
        max_token_ratio=0.3,
        p_apply=0.5,
    )

    return rec
