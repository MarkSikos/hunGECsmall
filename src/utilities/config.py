
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
random.seed(42)

API_BASE_URL = ""
API_KEY = ""
MODEL_NAME = ""
ANSWER_K = 4
MAX_EXAMPLES_FOR_NOW = 4000  
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

POOL_PATH = DATA_DIR / "error_task_pool.jsonl"
CLEAN_PATH = DATA_DIR / "clean_hu_sentences.jsonl"
INPUT_PATH = DATA_DIR / "clean_hu_sentences.jsonl"
OUTPUT_PATH = DATA_DIR / "gec_pairs_step2.jsonl"
TASK_POOL_PATH = DATA_DIR / "error_task_pool.jsonl"
TARGET_N_PAIRS = 3000

ANSWER_K = 4


def load_error_task_pool() -> List[Dict[str, Any]]:
    """
    Load error-generation tasks from TASK_POOL_PATH into memory.

    The path constant is expected to be defined elsewhere and imported
    into this module.
    """
    tasks: List[Dict[str, Any]] = []
    if TASK_POOL_PATH.exists():
        with TASK_POOL_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(json.loads(line))
    return tasks


ERROR_TASK_POOL = load_error_task_pool()

def sample_error_task_desc() -> Optional[str]:
    """
    Sample a random error-task description from the global pool.
    Returns None if the pool is empty.
    """
    if not ERROR_TASK_POOL:
        return None
    task = random.choice(ERROR_TASK_POOL)
    return task.get("description")

