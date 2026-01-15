from openai import OpenAI
from typing import Optional
from utilities.config import API_BASE_URL, API_KEY, MODEL_NAME

# Global OpenAI client configured from config
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def _strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from the model output and keep only
    the actual user-facing answer that follows them.
    """
    if "<think>" in text:
        # If closing tag exists, keep only the part after </think>
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        else:
            # If only <think> appears, just remove it
            text = text.replace("<think>", "").strip()
    return text.strip()


def call_llm(
    system: str,
    user: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Optional[str]:
    """
    Thin wrapper around the Chat Completions API:
    - sends system + user messages
    - disables reasoning mode
    - supports optional temperature / top_k / top_p
    - returns the first non-empty line of the cleaned response (or None)
    """
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # forward sampling controls when provided
    if top_k is not None:
        extra_body["top_k"] = top_k
    if top_p is not None:
        extra_body["top_p"] = top_p

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )

    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning_content", None)

    # in case the model only filled reasoning_content
    if (not content) and reasoning:
        content = str(reasoning).strip()

    if not content:
        return None

    # remove <think> blocks and keep the final answer
    content = _strip_think_blocks(content)
    first_line = content.split("\n", 1)[0].strip()
    return first_line or None
