import json
from config import DATA_DIR 

INPUT_PATH = DATA_DIR / "gec_pairs_step2_0.jsonl"
OUTPUT_PATH = DATA_DIR / "gec_sft.jsonl"


def make_prompt(incorrect: str) -> list[dict]:
    """
    Build TRL 'prompt' field: a single user turn with instruction + incorrect sentence.
    """
    content = (
        "Kérlek, javítsd ki a következő HIBÁS magyar mondatot "
        "nyelvtanilag helyes változatra, a jelentését megtartva.\n\n"
        f"Hibás mondat:\n{incorrect}"
    )
    return [{"role": "user", "content": content}]


def make_completion(correct: str) -> list[dict]:
    """
    Build TRL 'completion' field: a single assistant turn with the corrected sentence.
    """
    return [{"role": "assistant", "content": correct}]


def main() -> None:
    """
    Stream input JSONL, wrap each pair into (prompt, completion),
    and write TRL-conversational SFT records.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with INPUT_PATH.open(encoding="utf-8") as f_in, \
         OUTPUT_PATH.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            ex_id = ex.get("id")
            correct = ex["correct"].strip()
            incorrect = ex["incorrect"].strip()
            orig_meta = ex.get("meta", {})

            sft_record = {
                "id": ex_id,
                "prompt": make_prompt(incorrect),
                "completion": make_completion(correct),
                # keep original information under meta for traceability
                "meta": {
                    "incorrect": incorrect,
                    "correct": correct,
                    "gen_meta": orig_meta,
                },
            }

            f_out.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved {n} SFT examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
