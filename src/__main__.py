import argparse
import sys

from llm_sdk import Small_LLM_Model  # type: ignore

from src.function_selector import select_and_call_function
from src.io_utils import (
    load_functions_definition,
    load_prompts,
    save_results
)
from src.models import FunctionCallResult
from src.vocab import load_vocab

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transilate natural-language prompts into function calls"
        "using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=f"Path to the funcitons definition file "
        f"(default: {DEFAULT_FUNCTIONS_DEFINITION})",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to the prompts file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to the output results file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def run_pipeline(
    functions_definition_path: str, input_path: str, output_path: str
) -> None:

    try:
        functions = load_functions_definition(functions_definition_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading functions definition: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        prompts = load_prompts(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading prompts: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    print("Loading model...")
    model = Small_LLM_Model()
    vocab = load_vocab(model)

    results: list[FunctionCallResult] = []

    for index, entry in enumerate(prompts, start=1):
        print(f"[{index}/ {len(prompts)}] Processing: {entry.prompt!r}")
        try:
            result = select_and_call_function(
                model, entry.prompt, functions, vocab
            )
            results.append(result)
        except ValueError as e:
            print(f" Skipped due to error: {e}", file=sys.stderr)

    save_results(output_path, results)
    print(f"\nDone. {len(results)}/{len(prompts)} prompt(s) succeeded.")
    print(f"Results written to: {output_path}")


def main() -> None:
    """Parse arguments and run the pipeline."""
    args = parse_args()
    try:
        run_pipeline(args.functions_definition, args.input, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted by user. exiting gracefull.", file=sys.stderr)


if __name__ == "__main__":
    main()
