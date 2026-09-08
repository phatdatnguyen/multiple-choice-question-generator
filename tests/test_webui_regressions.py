"""UI failures and export snapshots, without making API requests."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import documents  # noqa: E402
import llm  # noqa: E402
import quiz  # noqa: E402
import webui  # noqa: E402


def table(stem="Câu hỏi về tiếng Việt?"):
    return pd.DataFrame([[stem, "Một", "Hai", "Ba", "Bốn", "B"]], columns=quiz.DATAFRAME_COLUMNS)


def generate(files, count=1, extra=""):
    return webui.on_generate_questions(
        "gpt-4o", "Vietnamese", count, extra, files, progress=lambda *args, **kwargs: None,
    )


def test_preparation_error_preserves_read_warnings_and_cause(monkeypatch):
    monkeypatch.setattr(documents, "build_document_context", lambda files: ("text", ["Unreadable old.doc"]))
    error = RuntimeError("Cannot tokenize")
    error.__cause__ = ValueError("Invalid encoding")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(documents, "chunk_text", fail)
    dataframe, update, notes = generate(["source.txt"])
    assert dataframe.empty and update["interactive"] is False
    assert "Unreadable old.doc" in notes
    assert "Cannot tokenize" in notes and "Invalid encoding" in notes


def test_missing_key_remains_visible_below_table(monkeypatch):
    monkeypatch.setattr(documents, "build_document_context", lambda files: ("text", []))

    def fail():
        raise llm.MissingAPIKey("Set OPENAI_API_KEY")

    monkeypatch.setattr(webui, "get_client", fail)
    dataframe, update, notes = generate(["source.txt"])
    assert dataframe.empty and update["interactive"] is False
    assert "Set OPENAI_API_KEY" in notes


def test_empty_generation_has_a_persistent_explanation(monkeypatch):
    monkeypatch.setattr(documents, "build_document_context", lambda files: ("text", []))
    monkeypatch.setattr(webui, "get_client", lambda: object())
    monkeypatch.setattr(llm, "generate_questions", lambda *args: [])
    dataframe, update, notes = generate(["source.txt"])
    assert dataframe.empty and update["interactive"] is False
    assert "No questions were generated" in notes


@pytest.mark.parametrize("count", [None, 0, 101, 1.5, float("inf")])
def test_invalid_question_count_is_reported_before_reading_files(count):
    dataframe, update, notes = generate(["not-read.txt"], count=count)
    assert dataframe.empty and update["interactive"] is False
    assert "whole number from 1 to 100" in notes


def test_generation_includes_extra_instructions_in_document_budget(monkeypatch):
    monkeypatch.setattr(documents, "build_document_context", lambda files: ("text", []))
    monkeypatch.setattr(webui, "get_client", lambda: object())
    monkeypatch.setattr(llm, "generate_questions", lambda *args: [quiz.Question("Q?", ["a", "b", "c", "d"], 0)])
    seen = []

    def budget(model, count, *, prompt_tokens):
        seen.append(prompt_tokens)
        return 100

    monkeypatch.setattr(llm, "document_token_budget", budget)
    generate(["source.txt"])
    generate(["source.txt"], extra="Hướng dẫn bổ sung. " * 100)
    assert seen[1] > seen[0] + 100


def test_generation_reports_document_batches_omitted_by_low_question_count(monkeypatch):
    monkeypatch.setattr(documents, "build_document_context", lambda files: ("text", []))
    monkeypatch.setattr(documents, "chunk_text", lambda *args: ["first", "second", "third"])
    monkeypatch.setattr(webui, "get_client", lambda: object())
    monkeypatch.setattr(llm, "generate_questions", lambda *args: [quiz.Question("Q?", ["a", "b", "c", "d"], 0)])
    _, _, notes = generate(["source.txt"])
    assert "Only the first 1 of 3 document batches" in notes


def test_repeated_export_names_keep_independent_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "EXPORT_DIR", str(tmp_path / "exports"))
    _, first = webui.on_export_questions(table("First question?"), "questions", "Aiken (.txt)")
    first_text = Path(first["value"]).read_text(encoding="utf-8")
    _, second = webui.on_export_questions(table("Second question?"), "questions", "Aiken (.txt)")
    assert first["value"] != second["value"]
    assert Path(first["value"]).read_text(encoding="utf-8") == first_text
    assert "Second question?" in Path(second["value"]).read_text(encoding="utf-8")
    assert Path(first["value"]).name == Path(second["value"]).name == "questions.txt"


def test_export_recovers_when_exports_path_is_a_file(tmp_path, monkeypatch):
    blocked = tmp_path / "exports"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(webui, "EXPORT_DIR", str(blocked))
    monkeypatch.setattr(webui.tempfile, "tempdir", str(tmp_path))
    status, update = webui.on_export_questions(table(), "questions", "Aiken (.txt)")
    assert update["visible"] is True
    assert "using a temporary file" in status
    assert "Câu hỏi về tiếng Việt?" in Path(update["value"]).read_text(encoding="utf-8")


def test_export_reports_failure_when_temporary_folder_is_also_unwritable(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "EXPORT_DIR", str(tmp_path / "exports"))

    def fail(*args, **kwargs):
        raise PermissionError("Disk is read-only")

    monkeypatch.setattr(webui.tempfile, "mkdtemp", fail)
    status, update = webui.on_export_questions(table(), "questions", "Aiken (.txt)")
    assert status.startswith("❌")
    assert "Could not create the download" in status and "Disk is read-only" in status
    assert update["visible"] is False and update["value"] is None


def test_table_edits_enable_export_and_clear_stale_download():
    export, status, download = webui.on_table_change(table())
    assert export["interactive"] is True
    assert status == "" and download["value"] is None and download["visible"] is False
    invalid = table()
    invalid.loc[0, "Answer"] = ""
    export, _, _ = webui.on_table_change(invalid)
    assert export["interactive"] is False


def test_cleared_sample_size_reports_a_note_without_discarding_the_table():
    original = table()
    result, notes = webui.on_shuffle(original, True, False, True, None)
    assert result is original
    assert "Could not shuffle" in notes
