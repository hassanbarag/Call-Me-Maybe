from src.vocab import clean_token_text

NUMBER_CHARS: set[str] = set("0123456789.-")
BOOLEAN_LITERALS: list[str] = ["true", "false"]


def build_number_token_ids(vocab: dict[str, int]) -> set[int]:
    """Return the set of token IDs whose text
    is made only of number characters."""

    allowed_ids: set[int] = set()
    for token_str, token_id in vocab.items():
        text = clean_token_text(token_str)
        if text and all(char in NUMBER_CHARS for char in text):
            allowed_ids.add(token_id)
    return allowed_ids


def build_string_token_ids(vocab: dict[str, int]) -> set[int]:
    """Return the set of token IDs safe to use
    while generating a stirng value."""

    allowed_ids: set[int] = set()
    for token_str, token_id in vocab.items():
        text = clean_token_text(token_str)
        if text and '"' not in text:
            allowed_ids.add(token_id)
    return allowed_ids


def build_choice_token_ids(
    vocab: dict[str, int], remaining_suffixes: list[str]
) -> dict[int, str]:

    """Return token IDs that are a valid next
    step towards one of the choices."""

    allowed: dict[int, str] = {}
    for token_str, token_id in vocab.items():
        text = clean_token_text(token_str)
        if not text:
            continue
        for suffix in remaining_suffixes:
            if suffix.startswith(text):
                allowed[token_id] = text
                break
    return allowed
