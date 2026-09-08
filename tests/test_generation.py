"""Generation plumbing, verified against a fake OpenAI client."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

import llm  # noqa: E402
import quiz  # noqa: E402
import webui  # noqa: E402


def make_payload(count, prefix="Q"):
    return json.dumps({
        "questions": [
            {
                "question": f"{prefix}{i}?",
                "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
                "answer": "ABCD"[i % 4],
            }
            for i in range(count)
        ]
    })


class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeResponses:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    def __init__(self, replies):
        self.responses = FakeResponses(replies)


# --------------------------------------------------------------------------- #
# generate_questions
# --------------------------------------------------------------------------- #

def test_generate_questions_maps_the_schema_onto_questions():
    client = FakeClient([make_payload(3)])
    questions = llm.generate_questions(client, "gpt-4o", "prompt", "document")
    assert [q.text for q in questions] == ["Q0?", "Q1?", "Q2?"]
    assert [q.answer_label for q in questions] == ["A", "B", "C"]
    assert questions[0].options == ["a", "b", "c", "d"]


def test_generate_questions_requests_a_strict_json_schema():
    """Structured output is what stops fenced code blocks breaking the parse."""
    client = FakeClient([make_payload(1)])
    llm.generate_questions(client, "gpt-4o", "prompt", "document")
    text_format = client.responses.calls[0]["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] == llm.QUESTION_SCHEMA


def test_generate_questions_omits_sampling_params_for_reasoning_models():
    client = FakeClient([make_payload(1)])
    llm.generate_questions(client, "gpt-5.6-luna", "prompt", "document")
    call = client.responses.calls[0]
    assert "temperature" not in call and "top_p" not in call


def test_generate_questions_sends_sampling_params_for_chat_models():
    client = FakeClient([make_payload(1)])
    llm.generate_questions(client, "gpt-4o", "prompt", "document")
    assert client.responses.calls[0]["temperature"] == 1


def test_generate_questions_rejects_deep_research_models_before_requesting():
    client = FakeClient([make_payload(1)])
    with pytest.raises(ValueError, match="does not support the strict structured output"):
        llm.generate_questions(client, "o3-deep-research", "prompt", "document")
    assert client.responses.calls == []


def test_generate_questions_handles_an_empty_result():
    client = FakeClient([json.dumps({"questions": []})])
    assert llm.generate_questions(client, "gpt-4o", "prompt", "document") == []


# --------------------------------------------------------------------------- #
# The UI callback
# --------------------------------------------------------------------------- #

@pytest.fixture
def document(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("\n\n".join(f"Paragraph {i} about biochemistry." for i in range(30)), encoding="utf-8")
    return str(path)


def run_generate(monkeypatch, client, document_files, *, model="gpt-4o", count=4, language="English"):
    monkeypatch.setattr(webui, "get_client", lambda: client)
    return webui.on_generate_questions(model, language, count, "", document_files, progress=lambda *a, **k: None)


def test_generate_callback_fills_the_table_and_enables_export(monkeypatch, document):
    client = FakeClient([make_payload(4)])
    dataframe, export_update, notes = run_generate(monkeypatch, client, [document])
    assert list(dataframe.columns) == quiz.DATAFRAME_COLUMNS
    assert len(dataframe) == 4
    assert export_update["interactive"] is True
    assert notes == ""


def test_generate_callback_splits_a_long_document_into_batches(monkeypatch, document):
    """A document over the context limit is chunked instead of erroring."""
    monkeypatch.setattr(llm, "document_token_budget", lambda model, count, **kwargs: 40)
    # One question per reply, with a distinct stem so nothing is deduplicated.
    client = FakeClient([make_payload(1, f"P{i}-") for i in range(40)])
    dataframe, _, notes = run_generate(monkeypatch, client, [document], count=6)
    assert len(client.responses.calls) > 1
    assert "split into" in notes
    # Every batch's questions end up in the table.
    assert len(dataframe) == len(client.responses.calls)


def test_generate_callback_spreads_the_requested_count_over_batches(monkeypatch, document):
    monkeypatch.setattr(llm, "document_token_budget", lambda model, count, **kwargs: 40)
    client = FakeClient([make_payload(1)] * 20)
    run_generate(monkeypatch, client, [document], count=7)
    counts = [
        int(call["instructions"].split("exactly ")[1].split(" ")[0])
        for call in client.responses.calls
    ]
    assert sum(counts) == 7


def test_generate_callback_survives_one_failed_batch(monkeypatch, document):
    monkeypatch.setattr(llm, "document_token_budget", lambda model, count, **kwargs: 40)
    replies = [make_payload(2), RuntimeError("rate limited"), make_payload(2)]
    client = FakeClient(replies + [make_payload(2)] * 20)
    dataframe, export_update, notes = run_generate(monkeypatch, client, [document], count=6)
    assert len(dataframe) > 0
    assert export_update["interactive"] is True
    assert "rate limited" in notes


def test_generate_callback_reports_a_total_failure(monkeypatch, document):
    client = FakeClient([RuntimeError("model not found")])
    dataframe, export_update, notes = run_generate(monkeypatch, client, [document])
    assert dataframe.empty
    assert export_update["interactive"] is False
    assert "model not found" in notes


def test_generate_callback_recovers_from_invalid_json(monkeypatch, document):
    """A model that ignores the schema must not take down the whole run."""
    client = FakeClient(["```json\n{not valid}\n```"])
    dataframe, export_update, notes = run_generate(monkeypatch, client, [document])
    assert dataframe.empty
    assert export_update["interactive"] is False
    assert notes


def test_generate_callback_flags_duplicate_questions(monkeypatch, document):
    payload = json.dumps({
        "questions": [
            {"question": "Which enzyme catalyses this reaction?", "option_a": "a",
             "option_b": "b", "option_c": "c", "option_d": "d", "answer": "A"},
            {"question": "Which enzyme catalyses this reaction?", "option_a": "a",
             "option_b": "b", "option_c": "c", "option_d": "d", "answer": "B"},
        ]
    })
    _, _, notes = run_generate(monkeypatch, client := FakeClient([payload]), [document], count=2)
    assert client.responses.calls
    assert "duplicates" in notes


def test_generate_callback_notes_a_short_result(monkeypatch, document):
    client = FakeClient([make_payload(2)])
    _, _, notes = run_generate(monkeypatch, client, [document], count=10)
    assert "2 valid questions, 10 were requested" in notes


def test_generate_callback_reports_unreadable_files(monkeypatch, document, tmp_path):
    legacy = tmp_path / "old.doc"
    legacy.write_bytes(b"\xd0\xcf")
    client = FakeClient([make_payload(2)])
    dataframe, _, notes = run_generate(monkeypatch, client, [document, str(legacy)], count=2)
    assert len(dataframe) == 2
    assert "Word 97-2003" in notes


def test_generate_callback_needs_a_document(monkeypatch):
    dataframe, export_update, _ = run_generate(monkeypatch, FakeClient([]), [])
    assert dataframe.empty
    assert export_update["interactive"] is False


def test_generate_callback_uses_the_selected_language(monkeypatch, document):
    client = FakeClient([make_payload(2)])
    run_generate(monkeypatch, client, [document], count=2, language="Vietnamese")
    instructions = client.responses.calls[0]["instructions"]
    assert "câu hỏi trắc nghiệm" in instructions
    assert "2" in instructions


# --------------------------------------------------------------------------- #
# Shuffle and export callbacks
# --------------------------------------------------------------------------- #

def table(rows):
    return pd.DataFrame(rows, columns=quiz.DATAFRAME_COLUMNS)


def test_shuffle_callback_samples_the_table():
    rows = [[f"Q{i}?", "a", "b", "c", "d", "A"] for i in range(8)]
    dataframe, _ = webui.on_shuffle(table(rows), True, False, True, 3)
    assert len(dataframe) == 3


def test_shuffle_callback_keeps_answers_correct():
    rows = [[f"Q{i}?", "w", "x", "y", "z", "C"] for i in range(6)]
    dataframe, _ = webui.on_shuffle(table(rows), False, True, False, 0)
    for _, row in dataframe.iterrows():
        assert row[f"Option {row['Answer']}"] == "y"


def test_shuffle_callback_requires_an_action():
    original = table([["Q?", "a", "b", "c", "d", "A"]])
    dataframe, _ = webui.on_shuffle(original, False, False, False, 0)
    assert dataframe is original


def test_export_callback_writes_each_format(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "EXPORT_DIR", str(tmp_path / "exports"))
    rows = [["Q1?", "a", "b", "c", "d", "B"]]
    for export_format in quiz.EXPORT_FORMATS:
        status, update = webui.on_export_questions(table(rows), "out", export_format)
        assert status.startswith("✅")
        assert os.path.isfile(update["value"])


def test_export_callback_contains_a_traversal_attempt(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    monkeypatch.setattr(webui, "EXPORT_DIR", str(export_dir))
    _, update = webui.on_export_questions(
        table([["Q?", "a", "b", "c", "d", "A"]]), "../../../../pwned", "Aiken (.txt)"
    )
    written = os.path.abspath(update["value"])
    assert written.startswith(os.path.abspath(str(export_dir)))
    assert os.path.basename(written) == "pwned.txt"


def test_export_callback_refuses_an_empty_table():
    status, update = webui.on_export_questions(webui.empty_dataframe(), "out", "Aiken (.txt)")
    assert status.startswith("❌")
    assert update["visible"] is False
