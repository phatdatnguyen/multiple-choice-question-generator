# Multiple-choice question generator

A Gradio web UI that turns documents into multiple-choice questions using the
OpenAI API, and exports them in formats Moodle and other LMSs can import.

- Upload one or more documents (PDF, Word, PowerPoint, EPUB, HTML, Markdown, plain text)
- Generate questions in English or Vietnamese
- Review and edit the questions in an editable table
- Shuffle question order, shuffle options, or keep a random subset
- Export as Aiken, GIFT, Moodle XML or CSV

## Installation

You need an OpenAI API key - register at
[platform.openai.com](https://platform.openai.com/api-keys).

Clone the repo:

```
git clone https://github.com/phatdatnguyen/multiple-choice-question-generator
cd multiple-choice-question-generator
```

Create a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install the required packages:

```
pip install -r requirements.txt
```

Create a file called `.env` and fill it in your API key:

```
OPENAI_API_KEY=sk-your-key-here
```

The key is read from the `OPENAI_API_KEY` environment variable first, then from
`.env`, and finally from a legacy `api_key.py` if you have one from an earlier
version. All three are gitignored - never commit a key.

## Start the web UI

Run `start_webui.bat`, or:

```
python webui.py
```

Then open http://127.0.0.1:7860 in a browser.

### Using it

1. Pick a model. The dropdown lists known models; press **↻** to replace the
   list with the models your account can actually use, or type a model ID
   directly.
2. Choose how many questions you want and which language to write them in.
3. Optionally add extra instructions ("focus on chapter 3", "avoid definition
   questions"). The full prompt is visible under **Prompt preview**.
4. Upload your documents and press **Generate questions**.
5. Edit anything you want in the table - it is fully editable.
6. Optionally shuffle or sample under **Shuffle and sample**.
7. Pick an export format, then press **Export questions**. The file is written
   to `exports/` and offered as a download.

Notes and warnings (unreadable files, failed batches, questions that look like
duplicates) appear below the table.

Documents longer than the model's context window are split automatically and the
requested question count is spread across the batches.

## Troubleshooting

**"Connection error"** - this is the OpenAI SDK's catch-all for low-level
failures, and it is usually not a network problem. Notes under the table now
include the underlying cause, for example:

```
Batch 1 of 1 failed: APIConnectionError: Connection error. <- TypeError: process() takes no keyword arguments
```

That particular one is an incompatibility between `httpx2` 2.12.0 (a dependency
of `openai`) and the `Brotli` package (a dependency of `gradio`): httpx2 decodes
brotli responses by passing an `output_buffer_limit` keyword that `Brotli`'s
`process()` does not accept, so every brotli-encoded response fails. `build_client()`
in `llm.py` detects this and asks the server for gzip instead. If you upgrade
`httpx2` or `Brotli` later and the incompatibility is gone, the workaround turns
itself off.

**"Model not found"** - the model list in `llm.py` can go stale. Press **↻** next
to the dropdown to replace it with the models your account can use, or type an ID
in directly.

**"No text found ... needs OCR first"** - the PDF has no text layer, only page
images. Run it through OCR before uploading.

## Command line tools

`shuffle_aiken.py` shuffles, samples and converts existing quiz files:

```
python shuffle_aiken.py input.txt output.txt -n 20 --shuffle-answers
```

| Option | Meaning |
| --- | --- |
| `-n`, `--n-questions N` | Keep only N questions, chosen at random |
| `--shuffle-answers` | Shuffle the options within each question |
| `--no-shuffle-questions` | Keep the original question order |
| `--ignore-answers` | Input has no `ANSWER:` lines |
| `--format` | `aiken` (default), `gift`, `xml` or `csv` |
| `--seed N` | Reproducible shuffling |

Convert an Aiken file to Moodle XML, keeping the order:

```
python shuffle_aiken.py input.txt quiz.xml --no-shuffle-questions --format xml
```

Shuffle a file that has no answer key:

```
python shuffle_aiken.py practice.txt shuffled.txt --ignore-answers --shuffle-answers
```

## Development

```
pip install -r requirements-dev.txt
pytest
```

| File | Contents |
| --- | --- |
| `webui.py` | Gradio interface and callbacks |
| `llm.py` | API key handling, model registry, question generation |
| `documents.py` | Text extraction, token counting, chunking |
| `quiz.py` | Question model, quiz file formats, shuffling |
| `shuffle_aiken.py` | Command line front end for `quiz.py` |

`quiz.py` and `documents.py` have no dependency on the UI or the API, so they can
be used as a library and are covered by the tests.

## License

See `LICENSE.txt`.
