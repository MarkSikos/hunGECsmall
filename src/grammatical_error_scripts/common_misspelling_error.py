from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from utilities.llm_client import call_llm
from prompt_engineering.answer_engineering import _basic_stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MISSPELL_PATH = DATA_DIR / "wiki_misspellings.jsonl"

# Split into word / non-word tokens so punctuation is preserved.
WORD_SPLIT_RE = re.compile(r"\w+|\W+", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Dictionary-based misspellings
# ---------------------------------------------------------------------------

def _load_misspell_dict() -> dict[str, list[str]]:
    """
    Load wiki misspelling list as {right_lower: [wrong1, wrong2, ...]}.
    """
    mapping: dict[str, list[str]] = defaultdict(list)

    if not MISSPELL_PATH.exists():
        return mapping

    with MISSPELL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            wrong = ex.get("wrong", "").strip()
            right = ex.get("right", "").strip()
            if not wrong or not right:
                continue
            key = right.lower()
            mapping[key].append(wrong)

    return mapping


MISSPELL_DICT = _load_misspell_dict()


def apply_dictionary_misspellings(sentence: str, prob: float = 0.51) -> str:
    """
    Replace correct words with known misspellings with probability `prob`.

    - Only whole tokens are replaced.
    - Attempts to preserve initial capitalization.
    """
    if not sentence or not MISSPELL_DICT:
        return sentence

    tokens = WORD_SPLIT_RE.findall(sentence)
    if not tokens:
        return sentence

    new_tokens: list[str] = []

    for tok in tokens:
        # Only touch alphanumeric "word" tokens.
        if tok.strip() and tok[0].isalnum():
            key = tok.lower()
            cand_wrongs = MISSPELL_DICT.get(key)
            if cand_wrongs and random.random() < prob:
                wrong = random.choice(cand_wrongs)

                # Preserve leading capital if present.
                if tok[0].isupper():
                    wrong = wrong[:1].upper() + wrong[1:]

                new_tokens.append(wrong)
                continue

        # Fallback: keep original token.
        new_tokens.append(tok)

    return "".join(new_tokens)


# ---------------------------------------------------------------------------
# LLM-based rare / specialized error patterns
# ---------------------------------------------------------------------------

def apply_rare_errors(
    sentence: str,
    p_apply: float = 0.5,
    max_retries: int = 2,
) -> str:
    """
    Optionally apply rare Hungarian error types via LLM.

    With probability `p_apply` we try to modify the sentence; if all
    candidates fail simple similarity/length checks, we return the original.
    """
    if random.random() > p_apply:
        return sentence

    system = (
        "Te egy magyar nyelvű GEC adatgenerátor vagy, ritka nyelvtani hibákat "
        "viszel egy mondatba."
    )
    user = f"""
Az alábbi mondat helyes, formális magyar nyelven íródott:

HELYES MONDAT:
\"\"\"{sentence}\"\"\"


Feladatod:
- Ha LEHETSÉGES, vigyél bele 1–2 ritkább, speciális magyar hibát az alábbi TÍPUSOK közül:
  
- Ha összetett főnevet látsz, néha írd szét („buszmegálló” → „busz megálló”, „kerékpárút” → „kerékpár út”),
  vagy írd egybe a különírandót („nem tudom” → „nemtudom”, „jó volna” → „jóvolna”).

- Ha intézménynév vagy cím szerepel (pl. „Eötvös Loránd Tudományegyetem”, „Debreceni Egyetem”),
  néha írd kisbetűvel vagy elrontottan („eötvös loránd tudomány egyetem”, „Debreceni egyetem”).

- Ha j/ly-t tartalmazó szót látsz, ritkán cseréld fel őket:
  - „hely” → „hej”, „folyam” → „fojam”, „gally” → „gaj”,
  - „János” → „Lyános”, „folyik” → „fojik”.

- Néha rontsd el a helyhatározói ragokat:
  - -ban/-ben ↔ -ba/-be keverése („iskolában” → „iskolába” egy állandó helyzetre,
    vagy fordítva),
  - -tól/-től ↔ -ról/-ről keverése („Pestről jövök” → „Pesttől jövök”).

- Időnként rontsd el a -val/-vel hasonulást:
  - „kézzel” → „kézvel”, „szívvel” → „szívvel” helyett „szívvel” rosszul írva, vagy
    „Béla val” → „Bélával” helyett „Béla val”.

- Ronts el alany–állítmány egyeztetést:
  - egyes vs. többes szám („A gyerekek játszik az udvaron.”),
  - személyegyeztetés („Mi elmentem a boltba.”).

- Ikes igék ragozását is téveszd el:
  - „eszem, iszom, alszom” helyett „eszek, iszok, alszok”,
  - vagy fordítva, hétköznapi alak helyett túl „irodalmi” alakot írj rossz környezetben.

- Néha rontsd el az igeragokat és igeidőket:
  - jelen/múlt keverése („Tegnap elmegyek a boltba.”),
  - E/3 ↔ E/1 vagy T/3 keverése („Ő mondták.”).

- Időnként rontsd el az igekötő helyét:
  - szétválasztod, amikor egybe kéne („megírtam” → „meg írtam”),
  - vagy egybeírod, amikor tagadás miatt külön kellene („nem írtammeg” a „nem írtam meg” helyett).
  
- Számok és dátumok írását is elronthatod:
  - „2024-ben” → „2024ben”, „március 3-án” → „március 3.-án”,
  - „10 000” → „10000” vagy fordítva, következetlenül.

- Birtokos szerkezetekben néha keverd a többes/egyes alakokat:
  - „a fiú könyvei” → „a fiú könyvei” vagy „a fiúk könyveit” rossz egyeztetéssel,
  - „A szülők gyermekeik” → „A szülők gyermekeik vannak” típusú hibák.

- Ezeket a speciális hibákat csak akkor, ha a mondatban
  ténylegesen jelen van olyan elem, amin a hiba értelmezhető.

Fontos:
- CSAK olyan hibát kövess el, amelyik tényleg illik a mondatra
  (pl. ne ronts el tulajdonnevet, ha nincs is benne).
- Ha a mondatban NINCS olyan elem, amire ezen hibák bármelyike értelmesen alkalmazható lenne,
  akkor add vissza VÁLTOZATLANUL az eredeti mondatot.
- A mondat jelentése nagyjából maradjon felismerhető.
- Egyetlen hibás mondatot adj vissza, metakommentár nélkül.
""".strip()

    for _ in range(max_retries):
        resp = call_llm(system=system, user=user, max_tokens=128, temperature=0.7)
        if not resp:
            continue

        cand = (
            resp.strip()
            .split("\n")[0]
            .strip()
            .strip('"')
            .strip("„")
            .strip("”")
        )
        if not cand:
            continue

        sim, len_ratio, change_ratio = _basic_stats(sentence, cand)

        # Simple constraints: similarity, amount of change, and length.
        if not (0.4 <= sim <= 0.98):
            continue
        if not (0.02 <= change_ratio <= 0.4):
            continue
        if not (0.7 <= len_ratio <= 1.3):
            continue

        return cand

    # Fallback if no candidate passes.
    return sentence
