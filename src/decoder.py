import math

from llm_sdk import Small_LLM_Model  # type: ignore

from src.grammar import build_choice_token_ids
from src.vocab import clean_token_text

EOS_FALLBACK_LOGIT = -math.inf


def mask_logits(logits: list[float], allowed_ids: set[int]) -> list[float]:
    """Return a copy of logits where disallowed token ids are set to -inf."""
    return [
        value if token_id in allowed_ids else EOS_FALLBACK_LOGIT
        for token_id, value in enumerate(logits)
    ]


def select_best_token(masked_logits: list[float]) -> int:
    """Return the token id with the highest logit value."""
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])

    if masked_logits[best_id] == EOS_FALLBACK_LOGIT:
        raise ValueError(
            "No valid token available: all logits masked to -inf."
        )
    return best_id


def append_literal(
    model: Small_LLM_Model, prompt_ids: list[int], literal: str
) -> list[int]:

    literal_ids = model.encode(literal)[0].tolist()
    return prompt_ids + literal_ids


def find_all_token_ids_for_text(vocab: dict[str, int], text: str) -> set[int]:
    """Find every vocabulary token id whose
    cleaned text exactly equals text."""
    matches = {
        token_id
        for token_str, token_id in vocab.items()
        if clean_token_text(token_str) == text
    }

    if not matches:
        raise ValueError(f"No token found in vocabulary for text: {text!r}")
    return matches


def generate_number(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    allowed_number_ids: set[int],
    stop_ids: set[int],
    max_tokens: int = 20,
) -> tuple[list[int], str]:
    """Generate a JSON number value
    token-by-token using constrained decoding."""

    generated_text = ""
    working_ids = list(prompt_ids)
    combined_mask = allowed_number_ids | stop_ids

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(working_ids)
        masked = mask_logits(logits, combined_mask)
        next_id = select_best_token(masked)

        if next_id in stop_ids:
            if not generated_text:
                raise ValueError("Model stopped before generating any digits.")
            return working_ids, generated_text

        token_text = model.decode([next_id]).replace("Ġ", " ").strip()
        working_ids.append(next_id)
        generated_text += token_text

    raise ValueError(
        f"Number generation did not stop within {max_tokens} tokens:"
        f"{generated_text!r}"
    )


def _has_immediate_repetition(text: str, min_unit_length: int = 3) -> bool:
    for unit_length in range(min_unit_length, len(text) // 2 + 1):
        tail = text[-unit_length:]
        before_tail = text[-2 * unit_length:-unit_length]
        if tail == before_tail:
            return True
    return False


def generate_string(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    allowed_string_ids: set[int],
    quote_token_ids: set[int],
    max_tokens: int = 40,
) -> tuple[list[int], str]:
    """Generate a JSON string value
    (without surrounding quotes) token-by-token."""

    working_ids = list(prompt_ids)
    generated_text = ""
    combined_mask = allowed_string_ids | quote_token_ids

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(working_ids)
        masked = mask_logits(logits, combined_mask)
        next_id = select_best_token(masked)

        if next_id in quote_token_ids:
            return working_ids, generated_text.strip()

        token_text = model.decode([next_id]).replace("Ġ", " ")
        working_ids.append(next_id)
        generated_text += token_text

        if _has_immediate_repetition(generated_text):
            return working_ids, generated_text.strip()

    raise ValueError(
        f"String generation did not stop within {max_tokens} tokens:"
        f"{generated_text!r}"
    )


def generate_choice(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    candidates: list[str],
    vocab: dict[str, int],
) -> tuple[list[int], str]:
    """Generate text constrained to exactly
    match one of a fixed set of candidates."""

    working_ids = list(prompt_ids)
    generated_text = ""
    remaining = list(candidates)

    while True:
        remaining = [c for c in remaining if c.startswith(generated_text)]

        if not remaining:
            raise ValueError(
                f"None candidate matches generated prefix: {generated_text!r}"
            )

        if len(remaining) == 1 and generated_text == remaining[0]:
            break

        suffixes = [c[len(generated_text):] for c in remaining]
        allowed = build_choice_token_ids(vocab, suffixes)

        if not allowed:
            raise ValueError("No valid tokens to continue choice generation.")

        logits = model.get_logits_from_input_ids(working_ids)
        masked = mask_logits(logits, set(allowed.keys()))
        next_id = select_best_token(masked)

        working_ids.append(next_id)
        generated_text += allowed[next_id]

    return working_ids, generated_text
