import json

from llm_sdk import Small_LLM_Model  # type: ignore

from src.decoder import (
    append_literal,
    find_all_token_ids_for_text,
    generate_choice,
    generate_number,
    generate_string,
)
from src.grammar import (
    BOOLEAN_LITERALS,
    build_number_token_ids,
    build_string_token_ids,
)
from src.models import FunctionCallResult, FunctionDefinition


def build_context_prompt(
    prompt: str, functions: list[FunctionDefinition]
) -> str:
    """Build the natural-language context
    given to the model before generation."""

    lines = ["Available functions:"]
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")

    lines.append("\nExamples:")
    for fn in functions:
        for example_request, example_answer in _build_examples(fn):
            lines.append(f'- "{example_request}" -> {example_answer}')

    lines.append(f"\nQuestion: {prompt}")
    lines.append("Function call:")
    return "\n".join(lines)


def _build_examples(fn: FunctionDefinition) -> list[tuple[str, str]]:
    known_examples: dict[str, list[tuple[str, str]]] = {
        "fn_add_numbers": [("What is the sum of 10 and 5?", fn.name)],
        "fn_greet": [("Greet shrek", fn.name)],
        "fn_reverse_string": [("Reverse the string 'hello'", fn.name)],
        "fn_get_square_root": [("What is the square root of 16?", fn.name)],
        "fn_substitute_string_with_regex": [
            ("Replace all digits in 'a1b2c3' with HASH", fn.name),
            ("Replace all letters in 'go go' with STAR", fn.name),
            ("Replace all numbers in 'x9y8' with NUM", fn.name),
        ],
    }
    return known_examples.get(
        fn.name, [(f"Use {fn.name} on some input", fn.name)]
    )


def find_function_by_name(
    functions: list[FunctionDefinition], name: str
) -> FunctionDefinition:
    """Find a function definition by its exact name."""

    for fn in functions:
        if fn.name == name:
            return fn
    raise ValueError(f"No function definition found for name: {name!r}")


def generate_parameter_value(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    vocab: dict[str, int],
    param_type: str,
) -> tuple[list[int], int | float | str | bool]:
    """Generate a single parameter value, dispatching on its declared type."""

    if param_type in ("number", "integer"):
        number_ids = build_number_token_ids(vocab)
        comma_ids = find_all_token_ids_for_text(vocab, ",")
        brace_ids = find_all_token_ids_for_text(vocab, "}")
        stop_ids = comma_ids | brace_ids
        prompt_ids, text = generate_number(
            model, prompt_ids, number_ids, stop_ids
        )

        try:
            parsed_number = float(text)
        except ValueError as e:
            raise ValueError(f"Generate invalid number text: {text!r}") from e

        if param_type == "integer":
            return prompt_ids, int(parsed_number)
        return prompt_ids, parsed_number

    if param_type == "string":
        string_ids = build_string_token_ids(vocab)
        quote_ids = find_all_token_ids_for_text(vocab, '"')
        prompt_ids = append_literal(model, prompt_ids, '"')
        prompt_ids, text = generate_string(
            model, prompt_ids, string_ids, quote_ids
        )
        prompt_ids = append_literal(model, prompt_ids, '"')
        return prompt_ids, text

    if param_type == "boolean":
        prompt_ids, text = generate_choice(
            model, prompt_ids, BOOLEAN_LITERALS, vocab
        )
        return prompt_ids, text == "true"

    raise ValueError(f"Unsupported parameter type: {param_type!r}")


def select_and_call_function(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition],
    vocab: dict[str, int],
) -> FunctionCallResult:
    """Translate one natural-language prompt into a full function call."""
    if not functions:
        raise ValueError("No function definitions provided.")

    context_text = build_context_prompt(prompt, functions)
    prompt_ids = model.encode(context_text)[0].tolist()

    function_names = [fn.name for fn in functions]
    prompt_ids, chosen_name = generate_choice(
        model, prompt_ids, function_names, vocab
    )
    chosen_function = find_function_by_name(functions, chosen_name)

    prompt_ids = append_literal(model, prompt_ids, "{")
    prompt_ids = append_literal(model, prompt_ids, json.dumps("name"))
    prompt_ids = append_literal(model, prompt_ids, ": ")
    prompt_ids = append_literal(model, prompt_ids, json.dumps(chosen_name))
    prompt_ids = append_literal(model, prompt_ids, ", ")
    prompt_ids = append_literal(model, prompt_ids, json.dumps("parameters"))
    prompt_ids = append_literal(model, prompt_ids, ": {")

    parameters: dict[str, int | float | str | bool] = {}
    param_items = list(chosen_function.parameters.items())

    for index, (param_name, param_schema) in enumerate(param_items):
        prompt_ids = append_literal(model, prompt_ids, json.dumps(param_name))
        prompt_ids = append_literal(model, prompt_ids, ": ")
        prompt_ids, value = generate_parameter_value(
            model, prompt_ids, vocab, param_schema.type
        )
        parameters[param_name] = value

        if index < len(param_items) - 1:
            prompt_ids = append_literal(model, prompt_ids, ", ")

    prompt_ids = append_literal(model, prompt_ids, "}")
    prompt_ids = append_literal(model, prompt_ids, "}")

    result = FunctionCallResult(
        prompt=prompt, name=chosen_name, parameters=parameters
    )
    data = {
        "name": result.name,
        "parameters": result.parameters,
    }
    print(f"Generating: {json.dumps(data)}")
    return result
