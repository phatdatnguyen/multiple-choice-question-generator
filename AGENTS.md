# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

The project uses a checked-in `venv/` (Python 3.10). On Windows:

```
venv\Scripts\activate
```

| Task | Command |
| --- | --- |
| Run the app | `python webui.py` (or `start_webui.bat`), then http://127.0.0.1:7860 |
| All tests | `pytest` (`pytest.ini` sets `testpaths = tests`) |
| One file | `pytest tests/test_quiz.py` |
| One test | `pytest tests/test_quiz.py::test_shuffle_options_keeps_correct_answer_with_duplicate_option_text` |
| Match by name | `pytest -k brotli` |
| Install dev deps | `pip install -r requirements-dev.txt` |
| CLI tool | `python shuffle_aiken.py input.txt output.txt -n 20 --shuffle-answers --format xml` |

There is no linter configured. `python -m pyflakes *.py tests/*.py` was used ad hoc; the tree is expected to stay clean under it.

The tests never touch the network — `tests/test_generation.py` sets a dummy `OPENAI_API_KEY` and drives the code through a `FakeClient`. Keep it that way; do not add tests that make live API calls.

## Architecture

Four modules with a strict dependency direction. `quiz.py` and `documents.py` are leaves that import nothing from the project, which is what makes them testable without Gradio or an API key:

```
webui.py  ──> llm.py ──> quiz.py
    └──────> documents.py
shuffle_aiken.py ──> quiz.py
```

- **`quiz.py`** — the `Question` model plus Aiken/GIFT/Moodle-XML/CSV serialisation, validation, shuffling, near-duplicate detection, and filename sanitisation.
- **`documents.py`** — text extraction per file type, content-hash caching, tiktoken counting, and token-budget chunking.
- **`llm.py`** — API key resolution, the model registry, prompt construction, the response schema, and generation.
- **`webui.py`** — Gradio components and callbacks only. No parsing or formatting logic belongs here.
- **`shuffle_aiken.py`** — thin CLI over `quiz.py`; it shares the same parse/shuffle/serialise code as the UI, so a fix in `quiz.py` reaches both.

## Invariants worth preserving

These encode bugs that were already fixed once.

**The correct answer is an index, never a letter.** `Question.answer_index` points into `Question.options`. Matching answers by option *text* silently mislabels the answer when two options have identical text, which is a common model output. `shuffle_options()` carries `(text, is_correct)` pairs through the shuffle for this reason.

**Question stems must be single-line.** Moodle's Aiken and GIFT importers reject multi-line stems, and models return newlines freely. `normalise_stem()` collapses whitespace and is applied in `validate()` and every serialiser.

**Never send `temperature`/`top_p` to a reasoning model.** The o-series and gpt-5 families reject them with a 400. `MODEL_REGISTRY` records which is which and `_request_kwargs()` is the only place that decides. Unknown model IDs deliberately default to `reasoning=True`: both values are the API defaults, so omitting them is a no-op for a chat model but avoids a hard failure on a reasoning one. Deep-research models additionally require a tool to be attached.

**The dropdown's model list is generated from `MODEL_REGISTRY`** (`MODEL_CHOICES = list(MODEL_REGISTRY)`). A second hardcoded list would drift; a test asserts they match. Model IDs go stale — the `↻` button calls `fetch_available_models()` to replace the list from the API.

**Model output shape is enforced by the API, not parsed from prose.** `QUESTION_SCHEMA` is sent as a strict `json_schema`. Strict mode does not support `minItems`/`maxItems`, hence the four explicit `option_a`…`option_d` fields, and `answer` is an `enum` of `A`–`D` so an unparseable answer is impossible. Do not relax this back to prompt-only instructions.

**The Brotli workaround in `build_client()` is load-bearing.** `httpx2` (an `openai` dependency) decodes brotli by passing an `output_buffer_limit` keyword that the `Brotli` package (a `gradio` dependency) rejects, and the SDK reports the resulting `TypeError` only as "Connection error". `brotli_decoding_is_broken()` probes the behaviour rather than checking versions, so the workaround disables itself once upstream is fixed. `DefaultHttpxClient` is used instead of a bare client to keep the SDK's 600-second timeout — a plain `httpx2.Client()` would drop to 5 seconds and break long generations.

**Export filenames go through `safe_file_stem()`.** The filename comes from a user-editable textbox and is interpolated into a path; without sanitisation `../../x` escapes `exports/`.

## Conventions

**Callbacks report, they don't raise.** UI callbacks return a `(dataframe, gr.update(...), notes)` tuple and surface problems two ways: `gr.Warning`/`gr.Info` toasts for immediate feedback, and a newline-joined `notes` string rendered under the table for detail. Validation and per-batch failures accumulate into `notes` so one bad batch or malformed question never discards the rest.

**Wrap exceptions with `llm.describe_error()`** when showing them, so the cause chain is visible rather than just the outermost message.

**Long documents are chunked, not truncated.** `document_token_budget()` decides how much text fits, `documents.chunk_text()` splits on paragraph then line boundaries, and `llm.split_count()` spreads the requested question count across batches.

**Prompts are built at request time** by `llm.build_prompt(language, count, extra_instructions)`. The UI textbox holds *extra* instructions only; the base prompt is never user-editable state that can drift from the slider. Add languages by extending `BASE_PROMPTS` — the dropdown reads its choices from it.

**Gradio 6 is required.** `Textbox(buttons=["copy"])` replaced `show_copy_button`, which no longer exists; `requirements.txt` carries a `gradio>=6.0` floor for this reason. Dependencies use lower bounds, not exact pins.

## API key

Resolved by `llm.resolve_api_key()` in order: `OPENAI_API_KEY` env var, then `.env` (parsed by a small built-in loader, no `python-dotenv` dependency), then a legacy `api_key.py` kept working for users of earlier versions. All three are gitignored.
