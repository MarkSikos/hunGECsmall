
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
from utilities.config import ANSWER_K  # exposed here if needed elsewhere

def _basic_stats(orig: str, corrupted: str) -> Tuple[float, float, float]:
    """
    Return (sim, len_ratio, change_ratio) based on token positions.

    - sim: ratio of identical tokens at the same positions
    - len_ratio: len(corrupted) / len(orig) in tokens
    - change_ratio: fraction of positions that changed
    """
    orig_toks = orig.split()
    corr_toks = corrupted.split()

    if not orig_toks or not corr_toks:
        return 0.0, 1.0, 1.0

    max_len = max(len(orig_toks), len(corr_toks))
    same_pos = 0
    for o, c in zip(orig_toks, corr_toks):
        if o == c:
            same_pos += 1

    sim = same_pos / max_len
    change_ratio = 1.0 - (same_pos / max_len)
    len_ratio = len(corr_toks) / max(len(orig_toks), 1)

    return sim, len_ratio, change_ratio


def score_noisy_candidate(
    orig: str,
    corrupted: str,
    max_change_ratio: float = 0.7,
) -> Tuple[float, float, float, float]:
    """
    Score a noisy candidate sentence.

    Higher score = better corruption: meaning mostly preserved,
    enough but not excessive changes.

    Returns: (score, sim, len_ratio, change_ratio)
    """
    corrupted = corrupted.strip()
    if not corrupted or corrupted == orig:
        return -1e9, 0.0, 1.0, 0.0

    sim, len_ratio, change_ratio = _basic_stats(orig, corrupted)

    # Too few or too many changes → reject hard
    if change_ratio < 0.05 or change_ratio > max_change_ratio * 1.5:
        return -1e9, sim, len_ratio, change_ratio

    # Meaning drifted too far (almost no overlap) → reject
    if sim < 0.3:
        return -1e9, sim, len_ratio, change_ratio

    # Length penalty
    length_penalty = abs(len_ratio - 1.0)

    # Target: change_ratio around ~70% of max_change_ratio
    target_change = max_change_ratio * 0.7
    change_penalty = abs(change_ratio - target_change)

    # Final score: balance similarity, amount of change, and length stability
    score = sim - 0.5 * length_penalty - 0.5 * change_penalty

    return score, sim, len_ratio, change_ratio


def choose_best_candidate(
    orig: str,
    candidate_texts: List[str],
    base_meta: Dict[str, Any],
    max_change_ratio: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Select the best corrupted candidate (if any) for a given original.

    candidate_texts:
        list of candidate corrupted sentences from one strategy/branch.
    base_meta:
        metadata for that branch (strategy, parameters, etc.) which
        will be extended with score/sim/len_ratio/change_ratio.

    Returns:
        {
          "incorrect": <best sentence>,
          "meta": { ... base_meta + score + sim + len_ratio + change_ratio ... }
        }
    or None if nothing passes the heuristics.
    """
    best_rec: Optional[Dict[str, Any]] = None
    best_score = -1e9

    for txt in candidate_texts:
        txt = txt.strip()
        if not txt or txt == orig:
            continue

        score, sim, len_ratio, change_ratio = score_noisy_candidate(
            orig, txt, max_change_ratio=max_change_ratio
        )
        if score <= -1e8:  # hard rejection
            continue

        if score > best_score:
            best_score = score
            meta = dict(base_meta)
            meta.update(
                {
                    "score": float(score),
                    "sim": float(sim),
                    "len_ratio": float(len_ratio),
                    "change_ratio": float(change_ratio),
                }
            )
            best_rec = {
                "incorrect": txt,
                "meta": meta,
            }

    return best_rec


