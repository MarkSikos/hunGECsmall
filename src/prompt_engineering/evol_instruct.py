from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from utilities.config import DATA_DIR, POOL_PATH, CLEAN_PATH
from utilities.llm_client import call_llm

random.seed(42)

# ---- basic thresholds and sampling hyperparameters ----
MIN_PASS_RATE_EVOLVED = 0.50
MIN_N_EVAL = 3
N_PARENTS_PER_RUN = 10          # how many parent tasks to sample from the pool
MIN_PARENT_PASS_RATE = 0.50     # only consider parents with at least this eval rate
MIN_PARENT_N_EVAL = 3

# ---- acceptance thresholds for different task types ----
MIN_PASS_RATE_SEED = 0.60       # stricter for hand-written seed tasks
MIN_PASS_RATE_EVOLVED = 0.50
MIN_N_EVAL = 3

# ---- sampling ----
N_PARENTS_PER_RUN = 10          # number of parents used for in-depth evolution per run
MIN_PARENT_PASS_RATE = 0.50     # minimum eval rate to be used as parent (if available)
MIN_PARENT_N_EVAL = 3
N_BREADTH_EXAMPLES = 8          # how many existing tasks to show as examples for breadth generation


# ------------ Data structure ------------

@dataclass
class ErrorTask:
    """
    Single error-generation task description stored in the pool.
    """
    id: str               # e.g. "seed_1", "depth_3", "breadth_5"
    description: str      # Hungarian task description (what to corrupt in a sentence)
    parent_id: str | None # ID of parent task (None for seeds)
    evol_type: str        # "seed", "in_depth", "in_breadth"
    eval_pass_rate: float
    n_eval: int


# ------------ Built-in seed task descriptions ------------

SEED_DESCRIPTIONS = [
    # 1 – helyesírási hibák
    "Készíts 3–4 helyesírási hibát a mondatban: ékezetek elhagyása, betűcserék, betűkihagyások. "
    "A szavak többsége maradjon változatlan, a mondat jelentése nagyjából ugyanaz maradjon.",

    # 2 – egyeztetés
    "Ronts el néhány nyelvtani egyeztetést: szám, személy, eset vagy igeidő szintjén. "
    "Legyen 2–4 nyelvtani hiba, de a mondat szerkezete nagyrészt maradjon az eredetihez hasonló.",

    # 3 – ragok, névutók
    "Keverj a mondatba rag- és névutó-hibákat: például -ban/-ben, -tól/-től, -hoz/-hez/-höz, "
    "illetve névutók (mellett, ellen, felé) téves használata. Legyen 2–3 ilyen hiba.",

    # 4 – szórend
    "Változtasd meg a mondat szórendjét úgy, hogy magyar anyanyelvű számára természetellenes, "
    "pongyola legyen, de még érthető maradjon. Legyen 1–2 szórendi hiba és legalább 4 helyesírási hiba.",

    # 5 – írásjelek
    "Hagyj ki vagy tegyél be vesszőket rossz helyre, és keverj bele 3–4 kisebb helyesírási hibát is. "
    "Írásjelekre és kötőszavakra fókuszálj.",
]


# ------------ String processing + pool helpers ------------

def sample_parent_tasks(tasks: list[ErrorTask], k: int) -> list[ErrorTask]:
    """
    Sample parent tasks for in-depth evolution, preferring tasks with
    enough evaluation data and good eval_pass_rate.
    """
    eligible = [
        t for t in tasks
        if t.n_eval >= MIN_PARENT_N_EVAL and t.eval_pass_rate >= MIN_PARENT_PASS_RATE
    ]
    if not eligible:
        # fallback: if nothing has eval info yet, use all tasks as potential parents
        eligible = tasks[:]

    if len(eligible) <= k:
        return eligible
    return random.sample(eligible, k)


def normalize_desc(text: str) -> str:
    """Lowercase + whitespace normalization for task descriptions."""
    return " ".join(text.lower().split())


def is_too_similar(new_desc: str, existing_descs: list[str], thresh: float = 0.9) -> bool:
    """
    Check if new_desc is too similar to any of the existing descriptions,
    using a simple character-based SequenceMatcher ratio.
    """
    for old in existing_descs:
        sim = SequenceMatcher(None, new_desc.lower(), old.lower()).ratio()
        if sim >= thresh:
            return True
    return False


def load_existing_pool() -> tuple[list[ErrorTask], list[str]]:
    """
    Load existing error_task_pool.jsonl (if present),
    and return:
      - tasks: list of ErrorTask objects
      - all_descs: list of all description strings
    """
    tasks: list[ErrorTask] = []
    all_descs: list[str] = []

    if not POOL_PATH.exists():
        return tasks, all_descs

    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = ErrorTask(
                id=obj["id"],
                description=obj["description"],
                parent_id=obj.get("parent_id"),
                evol_type=obj["evol_type"],
                eval_pass_rate=obj.get("eval_pass_rate", 0.0),
                n_eval=obj.get("n_eval", 0),
            )
            tasks.append(t)
            all_descs.append(t.description)

    return tasks, all_descs


def extract_ID(task_id: str) -> int:
    """
    Extract trailing integer from an ID (e.g. 'depth_17' -> 17),
    or 0 if no such suffix exists.
    """
    m = re.search(r"(\d+)$", task_id)
    if not m:
        return 0
    return int(m.group(1))


def get_next_ID_index(existing_tasks: list[ErrorTask]) -> int:
    """
    Find the maximum numeric suffix among existing task IDs
    and return the next free index (max + 1).
    """
    max_idx = 0
    for t in existing_tasks:
        n = extract_ID(t.id)
        if n > max_idx:
            max_idx = n
    return max_idx + 1


# ------------ Evaluation sentence sampling ------------

def load_eval_sentences(n: int = 10) -> list[str]:
    """
    Load some correct Hungarian sentences from clean_hu_sentences.jsonl
    to be used for automatic evaluation of tasks.
    """
    sents: list[str] = []
    with CLEAN_PATH.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            txt = ex["sentence"].strip()
            # simple length-based filter to avoid extreme cases
            if 5 <= len(txt.split()) <= 25:
                sents.append(txt)
            if len(sents) >= n * 3:
                break
    if len(sents) > n:
        sents = random.sample(sents, n)
    return sents


# ------------ LLM helper: apply a task to a single sentence ------------

def apply_task_to_sentence(sentence: str, task_description: str) -> str:
    """
    Use the LLM to corrupt a single Hungarian sentence according to
    the given error-generation task description.
    Returns exactly one (possibly corrupted) sentence.
    """
    system = "Te egy magyar nyelvű GEC adatgeneráló vagy."
    user = f"""
Az alábbi mondat helyes, formális magyar nyelven íródott:

MONDAT (HELYES):
\"\"\"{sentence}\"\"\"


Feladatod, hogy AZ ALÁBBI HIBA-GENERÁLÁSI LEÍRÁS szerint hibássá tedd:

HIBA-LEÍRÁS:
\"\"\"{task_description}\"\"\"


Követelmények:
- A mondat jelentése nagyjából maradjon felismerhető.
- Ne írj metakommentárt, csak az *egy* hibás mondatot add vissza.
- Ne tegyél hozzá új mondatot, csak EGYET adj vissza.
- A mondat hossza legyen kb. 70–130%-a az eredetinek.
"""
    resp = call_llm(system=system, user=user, max_tokens=128, temperature=0.9)
    if not resp:
        return sentence
    corrupted = resp.strip().split("\n")[0].strip()
    return corrupted


# ------------ Similarity heuristics on sentences ------------

def sentence_similarity(a: str, b: str) -> float:
    """Character-level similarity score between two sentences."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def changed_word_ratio(a: str, b: str) -> float:
    """
    Approximate ratio of changed words between two sentences.
    Very rough heuristic over whitespace-split tokens.
    """
    aw = a.split()
    bw = b.split()
    if not aw:
        return 1.0
    diff = sum(1 for x, y in zip(aw, bw) if x != y)
    diff += abs(len(aw) - len(bw))
    return diff / max(len(aw), len(bw))


# ------------ Automatic evaluation of evolved task descriptions ------------

def evaluate_task(task_description: str, n_samples: int = 6, debug: bool = False) -> tuple[float, int]:
    """
    Evaluate a task description by applying it to n_samples sentences
    and checking simple heuristic constraints:
      - output must differ from input
      - length ratio stays within [0.6, 1.6]
      - character similarity is not too low / not too high
      - some but not excessive word changes
    Returns (pass_rate, n_effective_samples).
    """
    sents = load_eval_sentences(n_samples)
    if not sents:
        return 0.0, 0

    passes = 0
    total = 0

    for sent in sents:
        corrupted = apply_task_to_sentence(sent, task_description)
        if not corrupted:
            continue

        total += 1

        sim = sentence_similarity(sent, corrupted)
        len_ratio = len(corrupted.split()) / max(1, len(sent.split()))
        change_ratio = changed_word_ratio(sent, corrupted)

        ok = True

        if corrupted.strip() == sent.strip():
            ok = False

        if not (0.6 <= len_ratio <= 1.6):
            ok = False

        if not (0.5 <= sim <= 0.99):
            ok = False

        if not (0.05 <= change_ratio <= 0.7):
            ok = False

        if ok:
            passes += 1

    if total == 0:
        return 0.0, 0

    return passes / total, total


# ------------ Evol-Instruct LLM calls (in-depth / in-breadth) ------------

def llm_generate_in_depth(parent_desc: str, n: int = 2) -> list[str]:
    """
    Generate in-depth variants of a given task description:
    same error family, but more detailed, with concrete examples/rules.
    The LLM is instructed to return a numbered list in Hungarian.
    """
    system = "Te egy adatgeneráló LLM vagy, aki GEC hiba-generáló feladatleírásokat finomít evol-instruct keretrendszerrel."
    user = f"""
Az alábbi hiba-generáló feladatleírást szeretném továbbfejleszteni (in-depth evolving):

\"\"\"{parent_desc}\"\"\"


Feladatod:
- Adj {n} darab új leírást, amelyek mind
  - ugyanebbe a hibacsaládba tartoznak,
  - részletesebbek, konkrét példákkal, pontosabb szabályokkal,
  - tisztán magyarul vannak megfogalmazva.

Kérlek, adj vissza pontosan {n} darab leírást, számozott listában, pl.:

1. ...
2. ...
"""
    resp = call_llm(system=system, user=user, max_tokens=512, temperature=0.8)
    if not resp:
        print("[IN-DEPTH] call_llm EMPTY response")
        return []

    text = resp.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    out: list[str] = []
    for l in lines:
        if l[0].isdigit():  # "1. ...."
            l = l.split(".", 1)[-1].strip()
        if len(l) > 20:
            out.append(l)

    out = out[:n]
    return out


def llm_generate_in_breadth(existing_descs: list[str], n: int = 5) -> list[str]:
    """
    Generate in-breadth variants: new types of error-generation tasks,
    inspired by (but not repeating) the existing descriptions.
    Again, the LLM returns a numbered list in Hungarian.
    """
    system = "Te egy adatgeneráló LLM vagy, aki magyar GEC hiba-generáló feladatleírásokat talál ki evol-instruct keretrendszerben."
    pool = existing_descs[:]
    random.shuffle(pool)
    examples = pool[:min(len(pool), N_BREADTH_EXAMPLES)]
    examples_text = "\n\n".join(f"- {d}" for d in examples)

    user = f"""
Ezek példák már létező hiba-generáló feladatleírásokra:

{examples_text}


Feladatod:
- Találj ki {n} új, ettől eltérő hibatípust leíró feladatot magyarul.
- Mindegyik legyen 2–4 mondatos, konkrét, gyakorlati útmutatás.
- Ne ismételd a fenti leírásokat, hanem egészítsd ki őket új típusú hibákkal (in breadth evolving)
  (pl. ikes igék, idegen szavak rossz írása, tulajdonnevek, dátumok, számok hibás írása stb.).

Kérlek, adj vissza pontosan {n} darab leírást, számozott listában, pl.:

1. ...
2. ...
3. ...
"""
    resp = call_llm(system=system, user=user, max_tokens=512, temperature=0.9)
    if not resp:
        print("[BREADTH] call_llm EMPTY response")
        return []

    text = resp.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    out: list[str] = []
    for l in lines:
        if l[0].isdigit():
            l = l.split(".", 1)[-1].strip()
        if len(l) > 20:
            out.append(l)

    out = out[:n]
    print(f"[BREADTH] {len(existing_descs)} meglévő desc → {len(out)} breadth jelölt")
    return out


# ------------ main pipeline ------------

def main():
    """
    Main entry point:
      - load existing pool
      - bootstrap with built-in seeds if pool is empty
      - generate in-depth variants from sampled parents
      - generate in-breadth variants as new error types
      - append only newly created tasks to the pool file
    """
    print("=== Evaluating SEED tasks ===")
    existing_tasks, all_descs = load_existing_pool()
    norm_set = {normalize_desc(d) for d in all_descs}
    next_idx = get_next_ID_index(existing_tasks)

    tasks: list[ErrorTask] = list(existing_tasks)
    new_tasks: list[ErrorTask] = []

    # 2) Evaluate built-in seed tasks only if there is no pool yet
    print("=== Evaluating SEED tasks ===")
    if existing_tasks:
        # Pool already exists → built-in seeds served only as bootstrap.
        print("Pool already exists -> skipping built-in SEED bootstrap.")
    else:
        for i, desc in enumerate(SEED_DESCRIPTIONS, start=1):
            norm = normalize_desc(desc)
            if norm in norm_set or is_too_similar(desc, all_descs, thresh=0.8):
                continue

            rate, n_eval = evaluate_task(desc, n_samples=4, debug=False)

            # Accept only if evaluation passes the threshold
            if n_eval < MIN_N_EVAL or rate < MIN_PASS_RATE_SEED:
                continue

            eid = f"seed_{next_idx}"
            next_idx += 1
            t = ErrorTask(
                id=eid,
                description=desc,
                parent_id=None,
                evol_type="seed",
                eval_pass_rate=rate,
                n_eval=n_eval,
            )
            tasks.append(t)
            new_tasks.append(t)
            all_descs.append(desc)
            norm_set.add(norm)

    # 3) IN-DEPTH variants from sampled parent tasks
    depth_tasks: list[ErrorTask] = []

    parents = sample_parent_tasks(tasks, N_PARENTS_PER_RUN)
    print("\n=== Generating in-depth variants ===")

    for t in parents:
        new_descs = llm_generate_in_depth(t.description, n=2)
        print(f"[IN-DEPTH] {t.id} parentből {len(new_descs)} jelölt jött")
        for d in new_descs:
            norm = normalize_desc(d)
            if norm in norm_set or is_too_similar(d, all_descs, thresh=0.8):
                print(455)
                continue

            rate, n_eval = evaluate_task(d, n_samples=6, debug=False)

            # Accept only if evaluation passes the threshold
            if n_eval < MIN_N_EVAL or rate < MIN_PASS_RATE_EVOLVED:
                print(461)
                continue

            eid = f"depth_{next_idx}"
            next_idx += 1

            dt = ErrorTask(
                id=eid,
                description=d,
                parent_id=t.id,
                evol_type="in_depth",
                eval_pass_rate=rate,
                n_eval=n_eval,
            )
            depth_tasks.append(dt)
            tasks.append(dt)
            new_tasks.append(dt)
            all_descs.append(d)
            norm_set.add(norm)

    # 4) IN-BREADTH variants – new error families
    print("\n=== Generating in-breadth variants ===")
    breadth_tasks: list[ErrorTask] = []
    breadth_descs = llm_generate_in_breadth(all_descs, n=5)

    print("\n=== Generating in-breadth variants ===")
    breadth_tasks: list[ErrorTask] = []
    breadth_descs = llm_generate_in_breadth(all_descs, n=5)

    for d in breadth_descs:
        norm = normalize_desc(d)
        if norm in norm_set or is_too_similar(d, all_descs, thresh=0.8):
            print(490)
            continue

        rate, n_eval = evaluate_task(d, n_samples=6, debug=False)
        if n_eval < MIN_N_EVAL or rate < MIN_PASS_RATE_EVOLVED:
            print(494)
            continue

        eid = f"breadth_{next_idx}"
        next_idx += 1
        print(f"[BREADTH {eid}] pass_rate={rate:.2f} (n={n_eval})")

        bt = ErrorTask(
            id=eid,
            description=d,
            parent_id="pool",
            evol_type="in_breadth",
            eval_pass_rate=rate,
            n_eval=n_eval,
        )
        breadth_tasks.append(bt)
        tasks.append(bt)
        new_tasks.append(bt)
        all_descs.append(d)
        norm_set.add(norm)

    # 5) Save – append only NEW tasks to the pool
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    if new_tasks:
        mode = "a" if POOL_PATH.exists() else "w"
        with POOL_PATH.open(mode, encoding="utf-8") as f:
            for t in new_tasks:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    print(f"\nPreviously existing tasks: {len(existing_tasks)}")
    print(f"Newly added tasks this run: {len(new_tasks)}")
    print(f"Total tasks in memory now: {len(tasks)}")
    if tasks:
        print("\nFirst 3 tasks (from full pool):\n")
        for t in tasks[:3]:
            print(f"- {t.id} [{t.evol_type}] rate={t.eval_pass_rate:.2f}: {t.description[:100]}...")


if __name__ == "__main__":
    main()
