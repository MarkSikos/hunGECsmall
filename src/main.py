from __future__ import annotations

import json
import random
from typing import Optional, Dict, Any, List
from utilities.config import INPUT_PATH, OUTPUT_PATH, TARGET_N_PAIRS, ANSWER_K, sample_error_task_desc
from grammatical_error_scripts.prompt_generated_error import generate_few_shot_noisy
from prompt_engineering.answer_engineering import  choose_best_candidate
from grammatical_error_scripts.typo_error import add_post_char_noise
from grammatical_error_scripts.common_misspelling_error import apply_dictionary_misspellings, apply_rare_errors
from grammatical_error_scripts.dependency_based_error import PARALEL_PAIRS, make_paralel_variant

##############################################################################################
#                                 DATA GENERATION PIPELINE  
##############################################################################################


def data_generation_pipeline(correct: str) -> Optional[Dict[str, Any]]:
    """
    Generate a corrupted variant for a single correct sentence.

    Branches:
      - ~5%: use pre-generated cross-lingual (paralel) variants, then optional char-noise
      - ~50%: few_shot + taskpool (specialized error types), best-of-k + optional char-noise
      - ~45%: generic few_shot (no taskpool), best-of-k + optional char-noise

    Returns:
      {
        "correct": gold_correct_sentence,
        "incorrect": corrupted_sentence,
        "meta": { ... }
      }
    or None if no reasonable corruption could be generated.
    """
    orig_correct = correct.strip()
    if not orig_correct:
        return None

    toks = orig_correct.split()
    if len(toks) < 5:
        return None

    r = random.random()

    # 0. Branch: small probability of paralel cloning
    if r < 0.05 and PARALEL_PAIRS:
        paralel_res = make_paralel_variant()
        if paralel_res is not None:
            return paralel_res
        # If paralel fails, we gracefully fall back to monolingual branches.
    # From here on we stay in monolingual (hu-only) branches.
    
    # 1) Pre-noising: dictionary misspellings + rare LLM-based errors
    #    This only affects the "incorrect" path; the gold "correct" stays orig_correct.
    working = apply_dictionary_misspellings(orig_correct, prob=0.5)
    working = apply_rare_errors(working, p_apply=0.5)

    r2 = random.random()

    # ------------------------------------------------------------------
    # few_shot + taskpool branch (~50%)
    # ------------------------------------------------------------------
    if 0.05 <= r2 < 0.55:
        task_desc = sample_error_task_desc()
        target_errs = random.choice([8, 10, 12])
        max_change_ratio = 0.7

        candidates: List[str] = []
        for _ in range(ANSWER_K):
            noisy = generate_few_shot_noisy(
                working,
                target_errors=target_errs,
                max_change_ratio=max_change_ratio,
                extra_task_desc=task_desc,
            )
            if noisy:
                candidates.append(noisy)

        if not candidates:
            return None

        base_meta = {
            "strategy": "few_shot_llm_taskpool",
            "target_errors": target_errs,
            "max_change_ratio": max_change_ratio,
            "task_desc": task_desc,
        }

        best = choose_best_candidate(
            orig_correct,
            candidates,
            base_meta=base_meta,
            max_change_ratio=max_change_ratio,
        )
        if best is None:
            return None

        best["correct"] = orig_correct

        best = add_post_char_noise(
            orig=orig_correct,
            rec=best,
            max_change_ratio=max_change_ratio,
            max_token_ratio=0.3,
            p_apply=0.5,
        )
        return best

    # ------------------------------------------------------------------
    # Generic few_shot branch (~45%)
    # ------------------------------------------------------------------
    if r2 >= 0.55:
        target_errs = random.choice([8, 10, 12])
        max_change_ratio = 0.8

        candidates: List[str] = []
        for _ in range(ANSWER_K):
            noisy = generate_few_shot_noisy(
                working,
                target_errors=target_errs,
                max_change_ratio=max_change_ratio,
                extra_task_desc=None,
            )
            if noisy:
                candidates.append(noisy)

        if not candidates:
            return None

        base_meta = {
            "strategy": "few_shot_llm_generic",
            "target_errors": target_errs,
            "max_change_ratio": max_change_ratio,
        }

        best = choose_best_candidate(
            orig_correct,
            candidates,
            base_meta=base_meta,
            max_change_ratio=max_change_ratio,
        )
        if best is None:
            return None

        best["correct"] = orig_correct

        best = add_post_char_noise(
            orig=orig_correct,
            rec=best,
            max_change_ratio=max_change_ratio,
            max_token_ratio=0.3,
            p_apply=0.5,
        )
        return best

    # If none of the branches fire, give up on this sentence
    return None

# -------------------------------------------------------------------------
# MAIN: iterator over the corpus and writes GEC pairs
# -------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    n_written = 0
    print(f"GEC data generation pipeline has started...")

    with INPUT_PATH.open(encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        print(f"Reading from {INPUT_PATH}, writing to {OUTPUT_PATH}...")

        for line in fin:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            src_id = ex.get("id")
            correct_in_file = ex["sentence"]

            res = data_generation_pipeline(correct_in_file)
            if res is None:
                continue

            # If the branch defines its own correct (paralel case), use that;
            # otherwise keep the original sentence as gold.
            gold_correct = res.get("correct", correct_in_file)

            rec = {
                "id": n_written,          # local GEC dataset id (0..N-1)
                "correct": gold_correct,
                "incorrect": res["incorrect"],
                "meta": {
                    **res["meta"],
                    "source_input_id": src_id,  # original clean corpus id
                },
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 100 == 0:
                print(f"Generated {n_written} pairs...")

            if n_written >= TARGET_N_PAIRS:
                break

    print(f"Saved {n_written} GEC pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
