import json
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Load the vocabulary file
    and return a token-string to token-id mapping."""
    vocab_path = Path(model.get_path_to_vocab_file())
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    """Build the reverse mapping: token-id to token-string."""
    return {token_id: token_str for token_str, token_id in vocab.items()}


def clean_token_text(token_str: str) -> str:
    return token_str.replace("Ġ", " ")
