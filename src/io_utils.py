import json
from pathlib import Path
from pydantic import ValidationError

from src.models import (
    FunctionDefinition,
    PromptEntry,
    FunctionCallResult,
)


def load_functions_definition(path: str) -> list[FunctionDefinition]:
    """Load and validate the functions_definition.json file."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Functions definition file not found: {path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in functions definition file: {e}"
        ) from e

    try:
        return [FunctionDefinition(**item) for item in raw_data]
    except ValidationError as e:
        raise ValueError(
            f"Function definition file failed validation: {e}"
        ) from e


def load_prompts(path: str) -> list[PromptEntry]:
    """Load and validate the function_calling_tests.json file."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompts file: {e}") from e

    try:
        return [PromptEntry(**item) for item in raw_data]
    except ValidationError as e:
        raise ValueError(f"Prompts file failed validation: {e}") from e


def save_results(path: str, results: list[FunctionCallResult]) -> None:
    """Write the final function-call results to a JSON file."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise ValueError(f"Failed to write result to {path}: {e}") from e
