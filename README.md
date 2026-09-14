# 📞 Call Me Maybe

## Description

**Call Me Maybe** is a project that explores **function calling with Large Language Models** through constrained decoding (logit masking).

Given a natural-language prompt and a catalog of available functions, the system translates user intent into a structured JSON function call with typed arguments.

It uses **Qwen/Qwen3-0.6B**, a small fully local language model, to demonstrate how **constrained decoding** can guide token generation and produce schema-compliant structured outputs.

## ✨ Features

* **Constrained decoding** — uses grammar-based constraints and logit masking to control token generation.
* **Schema-driven function calling** — uses function definitions to guide function and argument generation.
* **Grammar-based generation** — converts the expected output structure into constraints for the decoder.
* **Structured JSON output** — produces function calls with typed arguments in a predictable format.
* **Pydantic validation** — validates input definitions and generated results against defined data models.
* **Local LLM inference** — runs the function-calling pipeline using a local language model.

## 🧠 How It Works

```text
Natural Language Prompt + Function Definitions
                    │
                    ▼
        [1] Function Selection
          Identifies the target function
                    │
                    ▼
        [2] Grammar Construction
       Defines the valid output structure
                    │
                    ▼
        [3] Constrained Decoding
       Logit masking restricts valid tokens
                    │
                    ▼
        [4] Structured Output
       Function name + typed arguments
                    │
                    ▼
           Valid JSON Function Call
```

The key idea is to **constrain the generation process itself** rather than relying only on the model to produce valid structured output.

At each decoding step, the grammar determines which tokens are allowed, while logit masking prevents invalid tokens from being selected.

## 📋 Requirements

* Python 3.10+
* `uv` installed

## 🚀 Installation

1. Install the dependencies:

   ```bash
   make install
   ```

   Uses `uv` with workspace support to install `numpy`, `pydantic`, and the local `llm_sdk` package.

2. Run the default pipeline:

   ```bash
   make run
   ```

3. Run with custom input/output files:

   ```bash
   uv run python -m src \
     --functions_definition <file> \
     --input <file> \
     --output <file>
   ```

### Example

**Input prompt** (`data/input/`):

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks"
}
```

**Function catalog** (provided as a JSON schema):

```json
{
  "name": "fn_substitute_string_with_regex",
  "description": "Replace all occurrences matching a regex pattern in a string.",
  "parameters": {
    "source_string": { "type": "string" },
    "regex": { "type": "string" },
    "replacement": { "type": "string" }
  },
  "returns": { "type": "string" }
}
```

**Structured output:**

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "function": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "aeiouAEIOU",
    "replacement": "*"
  }
}
```

## 📁 Project Structure

```text
Call Me Maybe/
├── src/                    # Main pipeline: prompt construction, parsing, CLI
├── llm_sdk/                # Internal workspace package: model loading, constrained decoding
├── data/
│   └── input/              # Input prompts and function definitions
├── Makefile                # Install, run, and environment setup
└── pyproject.toml          # Project metadata and workspace config
```

## 🛠️ Tech Stack

| Component            | Library                              |
| -------------------- | ------------------------------------ |
| Language model       | Qwen/Qwen3-0.6B via HuggingFace      |
| Constrained decoding | Custom `llm_sdk` (workspace package) |
| Output validation    | `pydantic >= 2.13`                   |
| Numerical ops        | `numpy >= 2.2`                       |
| Package manager      | `uv` (workspace mode)                |
| Linting / typing     | `flake8`, `mypy`                     |
