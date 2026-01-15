from __future__ import annotations

import random
from typing import Optional, Dict, Any, List

from utilities.llm_client import call_llm
from utilities.config import sample_error_task_desc 
from prompt_engineering.answer_engineering import _basic_stats, choose_best_candidate
from utilities.config import ANSWER_K


# ---------------------------------------------------------------------------
# Few-shot style noisy generation (one candidate)
# ---------------------------------------------------------------------------

def build_few_shot_prompt(
    sentence: str,
    target_errors: int = 12,
    max_change_ratio: float = 0.8,
    extra_task_desc: Optional[str] = None,
) -> str:
    """
    Build a few-shot-style Hungarian GEC corruption prompt,
    optionally conditioned on an extra task description.
    """
    base_rules = f"""
Te egy adatgeneráló modul vagy, amely egy magyar nyelvű GEC (grammatical error correction) rendszer
betanításához készít HIBÁS mondatokat.

Alapszabályok:
- Bemenet: egy helyes, nyelvtanilag korrekt magyar mondat.
- Kimenet: ugyanaz a mondat, de SZÁNDÉKOSAN elrontva.
- A mondat jelentése maradjon nagyrészt ugyanaz, maximum árnyalatnyit térhet el.
- Ne írd át teljesen más szórenddel vagy szinonimákkal, csak finoman módosítsd.
- Kizárólag EGYETLEN mondatot adj vissza, semmilyen magyarázat nélkül.

Keverhető hibák:
- helyesírási hibák (betűkihagyás, betűcsere, hiányzó ékezetek),
- ragozási / egyeztetési hibák (szám, eset, személy, igeidő),
- rossz szórend, zavart okozó határozók vagy kötőszavak,
- elrontott jelek és ragok (-ban/-ben, -tól/-től, -hoz/-hez/-höz stb.),
- írásjelek hibái (hiányzó vessző, rossz pontozás).
- egybe írt szavak (kihagyott szóköz) vagy külön írt szavak (összevonás hiánya).

Korlátozások:
- Legyen körülbelül {target_errors} külön nyelvi hiba a mondatban.
- A szavak legfeljebb kb. {int(max_change_ratio*100)}%-át módosítsd (hibásítsd vagy mozgasd át).
- Ne változtasd meg a mondat hosszát drasztikusan (kb. hasonló szószám maradjon).
- Ne adj hozzá új mondatot, ne listázz, ne kommentálj.
- Ellenőrizd hogy a mondat valóban nyelvtanilag hibás legyen, ne csak a jelentéstartalom változzon. 
- Amennyiben nem nyelvtanilag hibás, csak a jelentés módosult: vigyél bele extra hibákat.
"""

    extra = ""
    if extra_task_desc is not None:
        extra = f"""

További speciális instrukciók ehhez a példához (ezeknek adj elsőbbséget a hibák kiválasztásakor):
\"\"\"{extra_task_desc}\"\"\""""

    return f"""
{base_rules}
{extra}

Bemeneti mondat:
\"\"\"{sentence}\"\"\"

Kimenetként csak a hibás mondatot add meg:
""".strip()


def sampling_profile(mode: str) -> Dict[str, Any]:
    """
    Simple sampling presets for LLM decoding.
    """
    if mode == "conservative":
        return dict(temperature=0.5, top_k=20, top_p=0.9)
    if mode == "creative":
        return dict(temperature=1.0, top_k=50, top_p=0.95)
    return dict(temperature=0.7, top_k=40, top_p=0.9)


def generate_few_shot_noisy(
    sentence: str,
    target_errors: int = 7,
    max_change_ratio: float = 0.7,
    extra_task_desc: Optional[str] = None,
    max_retries: int = 4,
) -> Optional[str]:
    """
    Generate a single corrupted sentence via few-shot-style prompting.

    Basic similarity / change-ratio checks are applied inline;
    fine-grained scoring happens elsewhere.
    """
    for _ in range(max_retries):
        prompt = build_few_shot_prompt(
            sentence,
            target_errors=target_errors,
            max_change_ratio=max_change_ratio,
            extra_task_desc=extra_task_desc,
        )
        params = sampling_profile("creative")

        system_prompt = (
            "Te egy magyar nyelvű adatgeneráló modul vagy, hibás mondatokat "
            "készítesz egy GEC modellhez."
        )

        resp = call_llm(
            system=system_prompt,
            user=prompt,
            max_tokens=128,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
        )

        noisy = (resp or "").strip().strip('"').strip("„").strip("”")
        if not noisy or noisy == sentence:
            continue

        # Only a coarse filter – detailed scoring is done later.
        _, _, change_ratio = _basic_stats(sentence, noisy)
        if change_ratio < 0.03:
            continue

        return noisy

    return None


# ---------------------------------------------------------------------------
# EVOL branches (taskpool-based and generic) using generate_few_shot_noisy
# ---------------------------------------------------------------------------

def branch_evol_taskpool(correct: str) -> Optional[Dict[str, Any]]:
    """
    EVOL branch: use a sampled error task (task pool) and best-of-k selection.
    """
    task_desc = sample_error_task_desc()
    target_errs = random.choice([8, 10, 12])
    max_change_ratio = 0.7

    candidates: List[str] = []
    for _ in range(ANSWER_K):
        noisy = generate_few_shot_noisy(
            correct,
            target_errors=target_errs,
            max_change_ratio=max_change_ratio,
            extra_task_desc=task_desc,
        )
        if noisy:
            candidates.append(noisy)

    if not candidates:
        return None

    base_meta = {
        "strategy": "evol_llm_taskpool",
        "target_errors": target_errs,
        "max_change_ratio": max_change_ratio,
        "task_desc": task_desc,
    }

    return choose_best_candidate(
        correct,
        candidates,
        base_meta=base_meta,
        max_change_ratio=max_change_ratio,
    )


def branch_evol_generic(correct: str) -> Optional[Dict[str, Any]]:
    """
    EVOL branch: generic evolution (no explicit task description), best-of-k.
    """
    target_errs = random.choice([8, 10, 12])
    max_change_ratio = 0.8

    candidates: List[str] = []
    for _ in range(ANSWER_K):
        noisy = generate_few_shot_noisy(
            correct,
            target_errors=target_errs,
            max_change_ratio=max_change_ratio,
            extra_task_desc=None,
        )
        if noisy:
            candidates.append(noisy)

    if not candidates:
        return None

    base_meta = {
        "strategy": "evol_llm_generic",
        "target_errors": target_errs,
        "max_change_ratio": max_change_ratio,
    }

    return choose_best_candidate(
        correct,
        candidates,
        base_meta=base_meta,
        max_change_ratio=max_change_ratio,
    )
