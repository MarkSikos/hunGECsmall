from __future__ import annotations
import random
from typing import Dict, Any, Optional, List
from prompt_engineering.answer_engineering import score_noisy_candidate, choose_best_candidate
from utilities.config import ANSWER_K


def confuse_j_ly(token: str) -> str:
    """
    Swap j ↔ ly inside a token (first occurrence only), preserving case.
    """
    lower = token.lower()

    # ly -> j
    idx = lower.find("ly")
    if idx != -1:
        is_upper = token[idx].isupper()
        repl = "J" if is_upper else "j"
        return token[:idx] + repl + token[idx + 2 :]

    # j -> ly
    idx = lower.find("j")
    if idx != -1:
        is_upper = token[idx].isupper()
        repl = "Ly" if is_upper else "ly"
        return token[:idx] + repl + token[idx + 1 :]

    return token


def corrupt_token(token: str) -> str:
    """
    Apply a single char-level corruption (j/ly or swap/drop/accent) to one token.
    """
    if len(token) <= 2:
        return token

    # 20% chance of j/ly confusion if relevant
    if random.random() < 0.2 and ("j" in token.lower() or "ly" in token.lower()):
        return confuse_j_ly(token)

    ops = ["swap", "drop", "accent"]
    op = random.choice(ops)

    if op == "swap" and len(token) >= 3:
        i = random.randint(0, len(token) - 2)
        return token[:i] + token[i + 1] + token[i] + token[i + 2 :]

    if op == "drop":
        i = random.randint(0, len(token) - 1)
        return token[:i] + token[i + 1 :]

    if op == "accent":
        mapping = str.maketrans(
            "áéíóöőúüűÁÉÍÓÖŐÚÜŰ",
            "aeiooouuuAEIOOOUUU",
        )
        return token.translate(mapping)

    return token


def add_char_noise(sentence: str, max_token_ratio: float = 0.5) -> str:
    """
    Corrupt a random subset of tokens with char-level noise.

    max_token_ratio: upper bound on the fraction of tokens to corrupt.
    """
    toks = sentence.split()
    if not toks:
        return sentence

    n_to_corrupt = max(1, int(len(toks) * random.uniform(0.1, max_token_ratio)))
    idxs = random.sample(range(len(toks)), k=min(n_to_corrupt, len(toks)))

    for i in idxs:
        toks[i] = corrupt_token(toks[i])

    return " ".join(toks)


def add_post_char_noise(
    orig: str,
    rec: Dict[str, Any],
    max_change_ratio: float,
    max_token_ratio: float = 0.15,
    p_apply: float = 0.3,
) -> Dict[str, Any]:
    """
    Optionally add extra char-level noise to an already selected candidate.

    - With probability p_apply it tries to add noise to rec["incorrect"].
    - If the new candidate scores too poorly, the original record is kept.
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


# Backwards-compatible alias for older code paths
def maybe_add_post_char_noise(
    orig: str,
    rec: Dict[str, Any],
    max_change_ratio: float,
    max_token_ratio: float = 0.15,
    p_apply: float = 0.3,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around add_post_char_noise.
    """
    return add_post_char_noise(
        orig=orig,
        rec=rec,
        max_change_ratio=max_change_ratio,
        max_token_ratio=max_token_ratio,
        p_apply=p_apply,
    )


def branch_char_noise(correct: str) -> Optional[Dict[str, Any]]:
    """
    Pure char-noise branch: generate candidates by char corruption only,
    then select the best one with the generic scoring heuristic.
    """
    max_token_ratio = 0.5
    candidates: List[str] = []
    for _ in range(ANSWER_K):
        noisy = add_char_noise(correct, max_token_ratio=max_token_ratio)
        if noisy and noisy != correct:
            candidates.append(noisy)

    if not candidates:
        return None

    base_meta = {
        "strategy": "char_noise_only",
        "max_token_ratio": max_token_ratio,
    }

    return choose_best_candidate(
        correct,
        candidates,
        base_meta=base_meta,
        max_change_ratio=0.5,
    )
