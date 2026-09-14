*This project has been created as part of the 42 curriculum by <hassamoh>.*

# Call Me Maybe — Function Calling with Constrained Decoding

## Description

This project implements a **function-calling engine** that translates natural-language
prompts into structured, schema-compliant function calls — without relying on a large,
general-purpose model to "just get it right."

Given a prompt like *"What is the sum of 2 and 3?"*, the goal is not to answer `5`, but to
produce:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The twist: the model used is **Qwen/Qwen3-0.6B**, a 500-million-parameter model that is,
on its own, notoriously unreliable at producing valid structured output. Simply prompting
it and hoping for well-formed JSON succeeds only a fraction of the time.

The solution is **constrained decoding**: instead of trusting the model to generate valid
JSON, the program inspects the model's raw output probabilities (logits) at every single
token, and mathematically forbids any token that would break JSON syntax or violate the
function schema — before the model is ever allowed to "choose" it. The result is JSON
that is **guaranteed to be syntactically valid and schema-compliant**, regardless of how
small or unreliable the underlying model is.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Internet access on first run (to download the Qwen3-0.6B model from Hugging Face)

### Installation

```bash
make install
# equivalent to: uv sync
```

This installs all dependencies (`pydantic`, `numpy`) and links the local `llm_sdk`
package, which wraps the Qwen3-0.6B model and exposes the low-level primitives
(`encode`, `decode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`) used to
drive constrained generation.

### Running

```bash
make run
# equivalent to: uv run python -m src
```

By default, this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`.

Custom paths can be supplied:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Makefile targets

```bash
make debug        # run under pdb
make lint         # flake8 + mypy (relaxed flags)
make lint-strict   # flake8 + mypy --strict
make clean         # remove __pycache__ and .mypy_cache
```

## Example Usage

```bash
$ make run
Loading model...
[1/2] Processing: 'What is the sum of 2 and 3?'
[2/2] Processing: 'Greet shrek'
Done. 2/2 prompt(s) succeeded.
Results written to: data/output/function_calling_results.json
```

Resulting `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Algorithm Explanation — Constrained Decoding

Language models generate text **one token at a time**. At every step, the model produces
a *logit* (a raw score) for every token in its vocabulary — for Qwen3-0.6B, that's
**151,936 possible tokens** at every single step. Normally, the token with the highest
score is picked, whatever it happens to be.

Constrained decoding intervenes **before** that selection happens:

1. **Compute logits** for the next token, given everything generated so far
   (`llm_sdk.get_logits_from_input_ids`).
2. **Determine which token ids are valid** at this exact position, based on what JSON
   syntax or schema type is expected right now (a digit? a quote? one of a fixed set of
   function names?).
3. **Mask every invalid token's logit to `-infinity`** (`decoder.mask_logits`), so it
   mathematically cannot be the maximum.
4. **Select the highest-scoring token among what remains** (`decoder.select_best_token`).
5. **Repeat**, feeding the chosen token back in as context for the next step.

Because invalid tokens are never even candidates, the output is **always** valid JSON —
the model cannot produce a stray comma, an unescaped quote, or a value of the wrong type,
no matter how small or "confused" it is.

### Building the vocabulary bridge (`vocab.py`)

The model only understands numeric token ids. To build a mask, the program needs a
`token string ↔ token id` mapping. This comes from the tokenizer's `vocab.json`
(`llm_sdk.get_path_to_vocab_file`), loaded once and kept as a simple two-way dictionary.
Token strings use a `Ġ` marker for a leading space (a BPE tokenizer convention), which is
normalized (`clean_token_text`) before any comparison.

### Building the constraint sets (`grammar.py`)

Three kinds of grammar-level masks are pre-computed once per generation:

- **Numbers**: every token whose text is made only of `0123456789.-`.
- **Strings**: every token that does *not* contain a `"` (the JSON string terminator).
- **Closed choices**: given a set of candidate full strings (e.g. function names, or
  `"true"`/`"false"`), the set of tokens that are a valid *prefix* of at least one
  remaining candidate — recomputed after every token, shrinking the candidate set until
  exactly one remains.

### The generation primitives (`decoder.py`)

- `generate_number` — generates digits until the model itself picks a "stop" token
  (`,` or `}`), which are included *in* the mask as a legitimate alternative to another
  digit. The stop token signals completion but is not appended to the value.
- `generate_string` — the same idea, using the closing quote `"` as the stop signal.
  Generation also aborts early if the text starts repeating an identical 4-character
  chunk — a well-documented failure mode of greedy decoding in small models, where the
  model loops on a fragment instead of finishing.
- `generate_choice` — used for function-name selection and booleans: at each step, only
  tokens that continue at least one remaining candidate are allowed, until one candidate
  is fully and uniquely generated.
- `append_literal` — for JSON syntax that is *always* required at a given position
  (`{`, `:`, `,`, a function name already chosen) — no model query happens here, the text
  is simply tokenized and appended, since there is nothing to "decide."

### Putting it together (`function_selector.py`)

For each prompt: the model is given a natural-language context listing the available
functions (with one example each) and the question, then:

1. `generate_choice` picks the function name from the closed set of available names.
2. JSON syntax is appended literally (`{"name": "...", "parameters": {`).
3. For each parameter, its type drives which generator is used (`generate_number` for
   `number`/`integer`, `generate_string` for `string`, `generate_choice` for `boolean`).
4. The JSON is closed, and the result is validated into a `FunctionCallResult` (pydantic).

## Design Decisions

- **Pydantic everywhere** (`models.py`): every data shape crossing a boundary (input
  files, output file, function schemas) is validated by a pydantic model, so malformed
  data fails loudly and early rather than causing a `KeyError` three functions later.
- **Separation of concerns**: `decoder.py` knows *nothing* about JSON schemas or function
  names — it only knows how to mask and select tokens. All schema-specific logic lives in
  `function_selector.py`, which composes the primitives. This kept each file small enough
  to reason about and test independently.
- **Stop tokens over guessed length limits**: early attempts capped generation length
  with a fixed `max_tokens`, which is fragile — a limit tuned for short names truncates
  longer values, and a generous limit lets the model ramble. Letting the model itself
  choose a designated "stop" token (comma, brace, quote) as an equally valid option at
  each step turned out to be the more principled solution.
- **Few-shot examples in the context prompt**: a bare list of function names and
  descriptions was not enough for the 0.6B model to disambiguate functions with
  overlapping vocabulary (e.g. a request containing numbers is not necessarily a math
  operation). Adding one short example per function meaningfully improved function
  selection accuracy.
- **Graceful, per-prompt error handling**: `select_and_call_function` raises `ValueError`
  on any failure. `__main__.py` catches this *per prompt* inside the processing loop, so
  one malformed or unusually hard prompt is skipped and reported, without aborting the
  entire batch. `KeyboardInterrupt` is caught separately at the top level for a clean
  exit.

## Performance Analysis

- **JSON validity: 100%.** Every prompt that completes produces syntactically valid,
  schema-compliant JSON — this is a hard guarantee of the masking approach, not a
  statistical outcome.
- **Function selection and simple argument extraction: consistently correct** across all
  arithmetic, greeting, and string-reversal prompts tested.
- **Speed**: a full batch of ~10 prompts completes in well under 5 minutes on standard
  hardware (CPU), comfortably within the project's requirement.
- **Known limitation — regex generation**: the `fn_substitute_string_with_regex` function
  asks the model to produce a `regex` pattern as one of its arguments. This is
  significantly harder than generating a number or a short name: the model must reason
  about *regex syntax itself*, not just copy or lightly transform text from the prompt.
  With a 0.6B model, this occasionally produces a regex with correct syntax (a valid
  character class or escape sequence) but that is not the *semantically tightest* pattern
  a human would write. Constrained decoding here guarantees the value cannot contain a
  raw space and must begin with `[` or `\` (steering it towards an actual pattern rather
  than a literal substring) and includes a repetition guard against the model looping —
  but it cannot force the model to *understand* what "all vowels" means as a regex
  concept. This is consistent with the subject's own framing (small language models are
  "notoriously unreliable" at structured, schema-heavy generation) and is the main source
  of any accuracy below 100% on the hardest prompts.

## Challenges Faced

- **Greedy decoding degenerate repetition**: early versions of `generate_string` would,
  on harder prompts, get the model stuck repeating an identical fragment indefinitely
  (e.g. a regex sub-pattern repeated dozens of times) instead of ever choosing the stop
  token. This is a documented failure mode of greedy decoding in small models. It was
  fixed by detecting an immediately-repeated character window and stopping generation as
  soon as it occurs, rather than trying to out-guess the model with token-length limits.
- **Stop-signal design for numbers and strings**: the first implementation relied on
  post-hoc checks (rejecting a token if it made the accumulated text invalid) rather than
  giving the model an explicit, always-available way to signal "I'm done." Including the
  actual stop tokens (`,`, `}`, `"`) *inside* the mask, as legitimate alternatives at
  every step, was a more robust fix than tuning length heuristics.
- **Context length vs. instruction-following**: instructions injected deep into a long,
  already-built prompt (e.g. "answer with only the value, nothing else") were sometimes
  *less* effective than expected, or even counter-productive, on a model this small —
  a reminder that small models have limited ability to hold and apply late instructions
  over a long context, and that structural constraints (masking) are more reliable than
  natural-language instructions for this model size.
- **`uv` and `huggingface_hub` cache locations**: on a shared/quota-limited home
  directory, both `uv`'s package cache and the Hugging Face model cache needed to be
  redirected (via `UV_CACHE_DIR` and `HF_HOME`) to a larger volume, since the ~1.5 GB
  model weights and PyTorch dependency exceeded the default home quota.

## Testing Strategy

- **Unit-level manual scripts** (not graded, used during development) exercised each
  module in isolation before integration: `vocab.py`/`grammar.py` token-id resolution,
  then `decoder.py`'s three generation primitives against hand-picked prompts, then the
  full `function_selector.py` pipeline.
- **End-to-end runs** against the full `function_calling_tests.json` /
  `functions_definition.json` pair, inspecting the resulting
  `function_calling_results.json` field by field against the expected function, argument
  names, and argument types.
- **Error-path testing**: missing input files, malformed JSON, and unsupported parameter
  types were each triggered deliberately to confirm the program reports a clear message
  and exits cleanly rather than crashing.
- **Repeated runs** to confirm determinism: since decoding is greedy (always the
  highest-logit token, never sampled), the same prompt reliably produces the same output
  across runs.

## Resources

- [Qwen3 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019)](https://arxiv.org/abs/1904.09751) — 
background on repetition/degeneracy in greedy decoding, referenced 
when designing the repetition guard in `generate_string`.
- [Byte-Pair Encoding tokenization overview](https://huggingface.co/docs/transformers/tokenizer_summary) — 
background on the `Ġ` leading-space marker used in `vocab.py`.
- Python [`re` module documentation](https://docs.python.org/3/library/re.html) — regex syntax reference.

### AI Usage Disclosure

An AI assistant (Claude) was used throughout this project as a **learning aid and
pair-programming guide**, not as a code-generation shortcut. Specifically, it was used
to:

- Explain unfamiliar concepts from the subject before any code was written — what
  function calling is, how LLM tokenization works, and what constrained decoding means
  mechanically (logits, masking, token-by-token generation).
- Walk through the design of the module structure 
All code in this repository was written and understood line by line by the author, who
can explain the purpose and behavior of any function on request, per the project's AI
usage guidelines.
