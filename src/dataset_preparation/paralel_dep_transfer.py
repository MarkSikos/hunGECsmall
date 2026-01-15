from __future__ import annotations
from typing import List, Tuple
from functools import lru_cache
import stanza


# ---------------------------------------------------------------------------
# Stanza pipelines (cached per language)
# ---------------------------------------------------------------------------

@lru_cache()
def get_hu_nlp():
    """
    Lazy-initialize Hungarian Stanza pipeline (tokenize + POS + lemma + deps).
    """
    stanza.download("hu", processors="tokenize,pos,lemma,depparse", verbose=False)
    return stanza.Pipeline(
        "hu",
        processors="tokenize,pos,lemma,depparse",
        use_gpu=False,
        tokenize_no_ssplit=True,
    )


@lru_cache()
def get_de_nlp():
    """
    Lazy-initialize German Stanza pipeline (tokenize + POS + lemma + deps).
    """
    stanza.download("de", processors="tokenize,pos,lemma,depparse", verbose=False)
    return stanza.Pipeline(
        "de",
        processors="tokenize,pos,lemma,depparse",
        use_gpu=False,
        tokenize_no_ssplit=True,
    )


# ---------------------------------------------------------------------------
# Helper functions: main verb + position category + token movement
# ---------------------------------------------------------------------------

def _find_main_verb(sent) -> int | None:
    """
    Return index (0-based) of the "main verb" in a Stanza Sentence:
      - VERB/AUX token with head=0 or deprel == "root".
    """
    candidates = []
    for i, w in enumerate(sent.words):
        if w.upos in ("VERB", "AUX"):
            if w.head == 0 or w.deprel == "root":
                candidates.append(i)
    if not candidates:
        return None
    return candidates[0]


def _verb_position_category(pos_idx: int, length: int) -> str:
    """
    Map verb index to a coarse position category: "initial", "middle", "final".
    """
    if length <= 1:
        return "middle"
    ratio = pos_idx / max(1, length - 1)
    if ratio <= 0.33:
        return "initial"
    elif ratio >= 0.66:
        return "final"
    else:
        return "middle"


def _move_token(tokens: List[str], idx: int, target_cat: str) -> List[str]:
    """
    Move tokens[idx] into the position slot specified by target_cat.
    """
    if idx < 0 or idx >= len(tokens) or len(tokens) <= 1:
        return tokens[:]

    toks = tokens[:]
    token = toks.pop(idx)

    if target_cat == "initial":
        toks.insert(0, token)
    elif target_cat == "final":
        toks.append(token)
    else:
        mid = len(toks) // 2
        toks.insert(mid, token)

    return toks


# ---------------------------------------------------------------------------
# Main error generator
# ---------------------------------------------------------------------------

def generate_paralel_syntax_error(
    de_sentence: str,
    hu_sentence: str,
) -> Tuple[str, list[str]] | None:
    """
    Given a German–Hungarian pair, generate a "foreign-sounding" Hungarian sentence.
    Returns:
        (incorrect_hu, error_types) or None if no transformation is applied.
    """
    de_sentence = de_sentence.strip()
    hu_sentence = hu_sentence.strip()

    if not de_sentence or not hu_sentence:
        return None

    hu_tokens = hu_sentence.split()
    if len(hu_tokens) < 5:
        return None

    nlp_de = get_de_nlp()
    nlp_hu = get_hu_nlp()

    de_doc = nlp_de(de_sentence)
    hu_doc = nlp_hu(hu_sentence)

    if not de_doc.sentences or not hu_doc.sentences:
        return None

    de_sent = de_doc.sentences[0]
    hu_sent = hu_doc.sentences[0]

    de_verb_idx = _find_main_verb(de_sent)
    hu_verb_idx = _find_main_verb(hu_sent)

    if de_verb_idx is None or hu_verb_idx is None:
        return None

    de_cat = _verb_position_category(de_verb_idx, len(de_sent.words))
    new_tokens = _move_token(hu_tokens, hu_verb_idx, de_cat)

    incorrect = " ".join(new_tokens)
    if incorrect == hu_sentence:
        return None

    error_types = ["paralel_syntax", "word_order"]
    return incorrect, error_types
