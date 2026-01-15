
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer, util

from utilities.llm_client import call_llm  # shared client used elsewhere

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IN_PATH = DATA_DIR / "raw_hu_sentences.jsonl"
OUT_PATH = DATA_DIR / "clean_hu_sentences.jsonl"

TARGET_TOTAL = 8000
MAX_SIM = 0.90  # cosine similarity threshold for near-duplicates


# ---------------------------------------------------------------------------
# Emoji / Cyrillic / "trash" heuristics
# ---------------------------------------------------------------------------

EMOJI_RE = re.compile(
    "["                    # broad emoji range
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")  # Cyrillic letters


def strip_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    return EMOJI_RE.sub("", text)


def looks_like_trash_or_non_hu(text: str) -> bool:
    """
    Fast heuristic for noisy / non-Hungarian text:
    - Cyrillic characters
    - too few letters
    - too many non-alphanumeric chars
    - URLs, hashtag / @ clusters, etc.
    Not a full lang detector, just filters obvious junk.
    """
    if not text or text.isspace():
        return True

    # Cyrillic → discard
    if CYRILLIC_RE.search(text):
        return True

    # basic whitespace normalization
    t = " ".join(text.strip().split())
    if not t:
        return True

    # too few alphabetic characters
    alpha_chars = [ch for ch in t if ch.isalpha()]
    if len(alpha_chars) < 5:
        return True

    # non-alphanumeric ratio
    non_alnum = sum(
        1 for ch in t if not (ch.isalnum() or ch.isspace())
    )
    ratio_non_alnum = non_alnum / max(1, len(t))
    if ratio_non_alnum > 0.40:
        return True

    # tokens
    tokens = re.findall(r"\w+", t, flags=re.UNICODE)
    if len(tokens) < 3:
        return True

    # strong URL / hashtag / @-handle signals
    if re.search(r"https?://|www\.", t, flags=re.I):
        return True
    if re.search(r"[@#]{2,}", t):  # e.g. "### ####"
        return True

    # majority of tokens 1–2 chars → likely junk
    short_tokens = sum(1 for tok in tokens if len(tok) <= 2)
    if short_tokens / len(tokens) > 0.6:
        return True

    return False


# ---------------------------------------------------------------------------
# LLM-based cleaning + quality / language filtering
# ---------------------------------------------------------------------------

def build_llm_filter_prompt(sentence: str) -> tuple[str, str]:
    """
    Build (system, user) prompts for the Hungarian LLM cleaning step.
    """
    system = (
        "You are a Hungarian text cleaner used for building a high-quality corpus. "
        "You must strictly follow the requested JSON output format."
    )

    user = f"""
Feladatod: egyetlen magyar mondatot kapsz. Döntsd el, hogy

- alkalmas-e egy igényes nyelvi korpuszba (hír, könyv, cikk, wikipedia-szerű szöveg, nincs benne trágárság),
- ha igen, javítsd a kisebb helyesírási hibákat, töröld az emojikat, szedd le a fölösleges szóismétlést, ha kisbetűvel kezdődik akkor írd nagybetűvel az elejét, ha kötőjellel vagy zárójellel vagy mondatvégi írásjellel kezdődik, akkor hagyd el a mondatot,
  de a mondat jelentése maradjon közel az eredetihez,
- ha a mondat főleg szemét: URL-halmaz, HTML-boilerplate, terméklista, spam, vagy nagyon rossz magyarság,
  akkor NE tartsd meg,
- HA A MONDAT NEM EGYÉRTELMŰEN MAGYAR (pl. angol, német, vegyes angol-magyar),
  AKKOR IS NE TARTSD MEG.

Kimenet:
- KIZÁRÓLAG EGY sor JSON-t adj vissza, semmi mást.
- NE használj markdown-t, kódblokkokat, kommentet.
- Formátum pontosan:

{{"keep": true/false, "fixed": "javított mondat vagy üres string"}}

Szabályok:
- Ha a mondat jó (csak kisebb hibák vannak), akkor "keep": true, és a "fixed" mezőbe írd a javított, emoji-mentes mondatot.
- Ha a mondat zajos / értelmetlen / NEM magyar / nem számmal vagy betűvel kezdődik /túl tele hibával, akkor "keep": false, a "fixed" legyen üres string.
- A "fixed" mindig egyetlen mondat legyen, új sor nélkül.

Eredeti mondat:
"{sentence}"
""".strip()

    return system, user


def extract_json_obj(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response:
      1) parse whole string
      2) parse content inside ```json ...``` or ```...```
      3) parse substring from first '{' to last '}'
    Raises ValueError on failure.
    """
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) fenced code blocks
    fence_match = re.search(r"```json(.*?)```", text, flags=re.S)
    if not fence_match:
        fence_match = re.search(r"```(.*?)```", text, flags=re.S)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass

    # 3) substring between first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(f"JSON parse failed from substring: {e}") from e

    raise ValueError("Could not find JSON in response.")


def llm_clean_and_filter(sentence: str):
    """
    Run LLM-based cleaning and filtering on a sentence.
    Returns: (keep: bool, cleaned_sentence: str | None).
    """
    # pre-clean: emoji strip + whitespace normalization
    pre = strip_emojis(sentence)
    pre = " ".join(pre.split())

    system, user = build_llm_filter_prompt(pre)

    try:
        resp = call_llm(
            system=system,
            user=user,
            max_tokens=128,
            temperature=0.4,   # conservative
            top_p=0.9,
            top_k=40,
        )
    except Exception as e:
        print(f"[LLM FILTER] call_llm failed, dropping: {e}")
        return False, None

    try:
        obj = extract_json_obj(resp)
    except Exception as e:
        print(f"[LLM FILTER] JSON parse error, dropping: {e}")
        return False, None

    keep = bool(obj.get("keep", False))
    fixed = obj.get("fixed", "")

    if not keep:
        return False, None

    if not isinstance(fixed, str) or not fixed.strip():
        print("[LLM FILTER] keep=true but empty fixed, dropping.")
        return False, None

    # final emoji strip + whitespace normalization
    fixed = strip_emojis(fixed)
    fixed = " ".join(fixed.split())

    return True, fixed


# ---------------------------------------------------------------------------
# I/O + SBERT-based semantic deduplication
# ---------------------------------------------------------------------------

def load_raw_by_source():
    """
    Load raw sentences and group them by 'source'.
    """
    by_source = defaultdict(list)
    with IN_PATH.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            by_source[ex["source"]].append(ex)
    return by_source


def main():
    """
    Balance per source, run heuristics + LLM filter,
    and perform SBERT-based deduplication.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ).to(device)

    by_source = load_raw_by_source()
    print("Sources and counts in raw file:")
    for src, lst in by_source.items():
        print(f"  {src}: {len(lst)} sentences")

    n_sources = len(by_source)
    target_per_source = TARGET_TOTAL // max(1, n_sources)
    print(f"Target per source ~{target_per_source}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_PATH.open("w", encoding="utf-8")

    global_embeddings = []
    kept_total = 0
    new_id = 0

    for src, examples in by_source.items():
        print(f"\nProcessing source '{src}' ...")

        local_kept = 0
        processed = 0
        dropped_heur = 0
        dropped_llm = 0
        dropped_dup = 0

        for ex in examples:
            if local_kept >= target_per_source:
                break

            sent = ex["sentence"]
            processed += 1

            # 0) Heuristic drop: Cyrillic / trash / too little signal
            if looks_like_trash_or_non_hu(sent):
                dropped_heur += 1
                continue

            # 1) LLM cleaning + quality filter
            keep, cleaned = llm_clean_and_filter(sent)
            if not keep or cleaned is None:
                dropped_llm += 1
                continue

            # 2) SBERT deduplication against global set
            emb = model.encode(cleaned, convert_to_tensor=True, device=device)
            if global_embeddings:
                sims = util.cos_sim(emb, torch.stack(global_embeddings))[0]
                if float(sims.max()) > MAX_SIM:
                    dropped_dup += 1
                    continue

            # accepted sentence
            global_embeddings.append(emb)
            local_kept += 1
            kept_total += 1

            rec = {
                "id": new_id,
                "sentence": cleaned,
                "source": src,
            }
            new_id += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if kept_total % 200 == 0:
                print(f"Kept {kept_total} sentences so far...")

        print(
            f"Source '{src}' done. "
            f"processed={processed}, kept={local_kept}, "
            f"dropped_heur={dropped_heur}, dropped_llm={dropped_llm}, "
            f"dropped_dup={dropped_dup}"
        )

    out_f.close()
    print(f"\nFinished. Kept {kept_total} sentences -> {OUT_PATH}")


if __name__ == "__main__":
    main()
