"""Gradio web UI for generating multiple-choice questions from documents."""

import os
import random
import tempfile

import gradio as gr
import pandas as pd

import documents
import llm
import quiz

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")

# The API key is resolved lazily so the UI can start and show a readable error
# instead of crashing on import.
_client = None


def get_client():
    global _client
    if _client is None:
        _client = llm.build_client()
    return _client


def empty_dataframe():
    return pd.DataFrame(columns=quiz.DATAFRAME_COLUMNS)


def questions_to_dataframe(questions):
    return pd.DataFrame(quiz.to_rows(questions), columns=quiz.DATAFRAME_COLUMNS)


def dataframe_to_questions(dataframe):
    """Read the (possibly hand-edited) table back into questions."""
    if dataframe is None:
        return [], []
    if isinstance(dataframe, pd.DataFrame):
        rows = dataframe.fillna("").values.tolist()
    else:
        rows = list(dataframe)
    return quiz.from_rows(rows)


def _format_notes(problems, prefix="⚠️"):
    return "\n".join(f"{prefix} {problem}" for problem in problems)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def on_preview_prompt(language, number_of_questions, extra_instructions):
    return llm.build_prompt(language, int(number_of_questions), extra_instructions)


def on_refresh_models(current_model):
    """Replace the model list with what the account can actually use."""
    try:
        names = llm.fetch_available_models(get_client())
    except Exception as exc:
        gr.Warning(f"Could not list models: {llm.describe_error(exc)}")
        return gr.update()
    if not names:
        gr.Warning("The API returned no usable chat models.")
        return gr.update()
    gr.Info(f"Found {len(names)} models available to your account.")
    value = current_model if current_model in names else names[0]
    return gr.update(choices=names, value=value)


def on_generate_questions(
    llm_model,
    language,
    number_of_questions,
    extra_instructions,
    document_files,
    progress=gr.Progress(),
):
    """Extract the documents, then generate questions in as many batches as needed."""
    notes = []

    def fail(message):
        gr.Warning(message)
        notes.append(message)
        return empty_dataframe(), gr.update(interactive=False), _format_notes(notes)

    if not document_files:
        return fail("Please upload at least one document first.")

    try:
        requested = int(number_of_questions)
        if requested != float(number_of_questions) or not 1 <= requested <= 100:
            raise ValueError("Choose a whole number from 1 to 100.")
    except (TypeError, ValueError, OverflowError):
        return fail("Number of questions must be a whole number from 1 to 100.")
    if not isinstance(llm_model, str) or not llm_model.strip():
        return fail("Please select a model first.")
    llm_model = llm_model.strip()

    progress(0.0, desc="Reading documents...")
    try:
        document_text, read_problems = documents.build_document_context(document_files)
    except Exception as exc:
        return fail(f"Could not read the documents: {llm.describe_error(exc)}")

    notes.extend(read_problems)
    if not document_text:
        return fail("None of the uploaded files could be read.")

    # Split the document if it will not fit in one request, and spread the
    # requested question count over the batches.
    try:
        prompt = llm.build_prompt(language, requested, extra_instructions)
        budget = llm.document_token_budget(
            llm_model, requested, prompt_tokens=documents.count_tokens(prompt, llm_model),
        )
        chunks = documents.chunk_text(document_text, budget, llm_model)
    except Exception as exc:
        return fail(f"Could not prepare the document: {llm.describe_error(exc)}")

    if not chunks:
        return fail("The documents contained no usable text.")

    if len(chunks) > 1:
        notes.append(
            f"Document exceeds the available {budget:,}-token input budget for "
            f"{llm_model}; split into {len(chunks)} batches."
        )

    per_chunk = llm.split_count(requested, len(chunks))
    if len(chunks) > requested:
        notes.append(
            f"Only the first {requested} of {len(chunks)} document batches will be used. "
            f"Request at least {len(chunks)} questions, use a model with a larger context, "
            "or upload smaller documents to cover all the text."
        )
    collected = []

    try:
        client = get_client()
    except llm.MissingAPIKey as exc:
        return fail(llm.describe_error(exc))
    except Exception as exc:
        return fail(f"Could not create the API client: {llm.describe_error(exc)}")

    for index, (chunk, count) in enumerate(zip(chunks, per_chunk)):
        if count <= 0:
            continue
        progress(
            index / len(chunks),
            desc=f"Generating batch {index + 1} of {len(chunks)} ({count} questions)...",
        )
        chunk_prompt = llm.build_prompt(language, count, extra_instructions)
        try:
            collected.extend(llm.generate_questions(client, llm_model, chunk_prompt, chunk))
        except Exception as exc:
            notes.append(f"Batch {index + 1} of {len(chunks)} failed: {llm.describe_error(exc)}")

    progress(1.0, desc="Checking results...")

    if not collected:
        return fail("No questions were generated. Try again or choose a different model.")

    questions, validation_problems = quiz.validate(collected)
    notes.extend(validation_problems)

    for i, j, similarity in quiz.find_near_duplicates(questions):
        notes.append(
            f"Questions {i + 1} and {j + 1} look like duplicates "
            f"({similarity:.0%} overlap): \"{questions[i].text[:60]}...\""
        )

    if not questions:
        return fail("Every generated question failed validation.")

    if len(questions) != requested:
        notes.append(f"Model returned {len(questions)} valid questions, {requested} were requested.")

    gr.Info(f"Generated {len(questions)} questions.")
    return (
        questions_to_dataframe(questions),
        gr.update(interactive=True),
        _format_notes(notes),
    )


def on_shuffle(question_dataframe, shuffle_order, shuffle_answers, sample_enabled, sample_size):
    """Shuffle and/or sample the table in place."""
    questions, problems = dataframe_to_questions(question_dataframe)
    if not questions:
        gr.Warning("There are no valid questions to shuffle.")
        return question_dataframe, _format_notes(problems)

    if not (shuffle_order or shuffle_answers or sample_enabled):
        gr.Warning("Select at least one of shuffle or sample.")
        return question_dataframe, _format_notes(problems)

    try:
        result = quiz.shuffle_quiz(
            questions,
            shuffle_order=bool(shuffle_order),
            shuffle_answers=bool(shuffle_answers),
            sample=int(sample_size) if sample_enabled else None,
            rng=random,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        problems.append(f"Could not shuffle the questions: {llm.describe_error(exc)}")
        gr.Warning(problems[-1])
        return question_dataframe, _format_notes(problems)

    gr.Info(f"{len(result)} questions ready.")
    return questions_to_dataframe(result), _format_notes(problems)


def on_table_change(question_dataframe):
    """Enable export for valid edits and invalidate downloads of earlier rows."""
    questions, _ = dataframe_to_questions(question_dataframe)
    return gr.update(interactive=bool(questions)), "", gr.update(value=None, visible=False)


def on_export_questions(question_dataframe, file_name, export_format):
    """Write the questions to disk and offer them as a download."""
    hidden = gr.update(value=None, visible=False)

    questions, problems = dataframe_to_questions(question_dataframe)
    if not questions:
        message = "Nothing to export - the table has no valid questions."
        return f"❌ {message}", hidden

    try:
        text, suffix = quiz.serialise(questions, export_format)
    except ValueError as exc:
        return f"❌ {exc}", hidden

    # Reduce the name to a bare filename so it cannot escape the export folder.
    stem = quiz.safe_file_stem(file_name)
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        # Each download keeps its own snapshot even if another export uses the
        # same name before Gradio has copied the file into its download cache.
        directory = tempfile.mkdtemp(prefix="mcq-", dir=EXPORT_DIR)
        target = os.path.join(directory, stem + suffix)
        with open(target, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
    except OSError as exc:
        # Still let the user download even if the local copy cannot be written.
        problems.append(
            f"Could not write to the exports folder ({llm.describe_error(exc)}); "
            "using a temporary file."
        )
        try:
            target = os.path.join(tempfile.mkdtemp(prefix="mcq-"), stem + suffix)
            with open(target, "w", encoding="utf-8", newline="\n") as file:
                file.write(text)
        except OSError as fallback_exc:
            problems.append(f"Could not create the download: {llm.describe_error(fallback_exc)}")
            return _format_notes(problems, prefix="❌"), hidden

    status = [f"✅ Exported {len(questions)} questions to `{target}`."]
    status.extend(f"⚠️ {problem}" for problem in problems)
    return "\n\n".join(status), gr.update(value=target, visible=True)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

DEFAULT_QUESTION_COUNT = 20

with gr.Blocks(title="Multiple-choice question generator") as demo:
    gr.Markdown("# Multiple-choice question generator")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                llm_model = gr.Dropdown(
                    label="Model",
                    value=llm.DEFAULT_MODEL,
                    choices=llm.MODEL_CHOICES,
                    scale=4,
                    allow_custom_value=True,
                )
                refresh_models_button = gr.Button("↻", scale=1, min_width=40)
            number_of_questions = gr.Slider(
                label="Number of questions", minimum=1, maximum=100, step=1,
                value=DEFAULT_QUESTION_COUNT,
            )
            language = gr.Dropdown(label="Language", value="English", choices=list(llm.BASE_PROMPTS))
            extra_instructions = gr.Textbox(
                label="Extra instructions (optional)",
                placeholder="e.g. focus on chapter 3, avoid definition questions, aim at undergraduates",
                lines=4,
            )
            with gr.Accordion("Prompt preview", open=False):
                prompt_preview = gr.Textbox(
                    label="Sent to the model",
                    value=llm.build_prompt("English", DEFAULT_QUESTION_COUNT),
                    lines=8,
                    interactive=False,
                    buttons=["copy"],
                )
            document_input = gr.File(
                label="Upload documents",
                file_count="multiple",
                file_types=documents.SUPPORTED_EXTENSIONS,
                type="filepath",
            )
            generate_button = gr.Button("Generate questions", variant="primary")

        with gr.Column(scale=3):
            question_dataframe = gr.Dataframe(
                headers=quiz.DATAFRAME_COLUMNS,
                datatype=["str"] * len(quiz.DATAFRAME_COLUMNS),
                column_count=(len(quiz.DATAFRAME_COLUMNS), "fixed"),
                wrap=True,
                label="Questions (editable)",
            )
            notes_output = gr.Markdown(value="", label="Notes")

            with gr.Accordion("Shuffle and sample", open=False):
                with gr.Row():
                    shuffle_order = gr.Checkbox(label="Shuffle question order", value=True)
                    shuffle_answers = gr.Checkbox(label="Shuffle options", value=False)
                with gr.Row():
                    sample_enabled = gr.Checkbox(label="Keep only", value=False)
                    sample_size = gr.Number(label="questions", value=10, precision=0, minimum=1)
                shuffle_button = gr.Button("Apply")

            with gr.Row():
                export_format = gr.Dropdown(
                    label="Format", choices=quiz.EXPORT_FORMATS, value=quiz.EXPORT_FORMATS[0],
                )
                file_name = gr.Textbox(label="File name", value="questions")
            export_button = gr.Button("Export questions", interactive=False)
            export_status = gr.Markdown(value="")
            export_file = gr.File(label="Download", visible=False, interactive=False)

    prompt_inputs = [language, number_of_questions, extra_instructions]
    for control in prompt_inputs:
        control.change(on_preview_prompt, prompt_inputs, prompt_preview)

    refresh_models_button.click(on_refresh_models, llm_model, llm_model)

    generate_button.click(
        on_generate_questions,
        [llm_model, language, number_of_questions, extra_instructions, document_input],
        [question_dataframe, export_button, notes_output],
    )

    shuffle_button.click(
        on_shuffle,
        [question_dataframe, shuffle_order, shuffle_answers, sample_enabled, sample_size],
        [question_dataframe, notes_output],
    )

    question_dataframe.change(
        on_table_change,
        question_dataframe,
        [export_button, export_status, export_file],
    )

    export_button.click(
        on_export_questions,
        [question_dataframe, file_name, export_format],
        [export_status, export_file],
    )


def main():
    try:
        llm.resolve_api_key()
    except llm.MissingAPIKey as exc:
        print(f"Warning: {exc}")
    demo.launch(max_file_size=100 * gr.FileSize.MB)


if __name__ == "__main__":
    main()
