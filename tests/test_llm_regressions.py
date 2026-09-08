"""Network-free regressions for credentials and generation failures."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
import quiz  # noqa: E402


def client_for(response):
    return SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))


def generate(response):
    return llm.generate_questions(client_for(response), "gpt-4o", "prompt", "document")


def test_resolve_api_key_does_not_read_dotenv_with_a_valid_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")

    def unreadable(*args):
        raise PermissionError("cannot read .env")

    monkeypatch.setattr(llm, "load_dotenv", unreadable)
    assert llm.resolve_api_key() == "sk-from-environment"


def test_resolve_api_key_finds_dotenv_next_to_the_app_from_another_directory(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    (app / ".env").write_text("OPENAI_API_KEY=sk-from-app", encoding="utf-8-sig")
    monkeypatch.setattr(llm, "__file__", str(app / "llm.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.resolve_api_key() == "sk-from-app"


def test_document_budget_accounts_for_long_instructions():
    original = llm.document_token_budget("gpt-4o", 20)
    assert llm.document_token_budget("gpt-4o", 20, prompt_tokens=10000) == original - 10000
    with pytest.raises(ValueError, match="instructions leave no room"):
        llm.document_token_budget("gpt-4o", 20, prompt_tokens=original)


@pytest.mark.parametrize(("model", "context"), [
    ("gpt-5-chat-latest", 128000), ("gpt-5.1-chat-latest", 128000),
    ("gpt-5.2-chat-latest", 128000), ("gpt-5.4-mini", 400000), ("gpt-5.4-nano", 400000),
])
def test_document_budget_uses_the_variant_context_window(model, context):
    # Chat and small variants do not inherit their larger sibling's context.
    assert llm.document_token_budget(model, 20, prompt_tokens=500) < context


@pytest.mark.parametrize("model", [
    "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0613",
    "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "o3-deep-research",
    "o3-deep-research-2025-06-26", "o4-mini-deep-research-2025-06-26",
])
def test_known_incompatible_models_fail_before_contacting_the_api(model):
    def unexpected_request(**kwargs):
        pytest.fail("An incompatible model must not receive a generation request")

    client = SimpleNamespace(responses=SimpleNamespace(create=unexpected_request))
    with pytest.raises(ValueError, match="does not support the strict structured output"):
        llm.generate_questions(client, model, "prompt", "document")
    assert model not in llm.MODEL_CHOICES


def test_refresh_models_excludes_incompatible_models_without_excluding_new_models():
    names = [
        "gpt-4-0613", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06",
        "o3-deep-research-2025-06-26", "gpt-4-turbo", "gpt-3.5-turbo-0125",
        "gpt-9-future", "gpt-6-astra", "gpt-4o-audio-preview", "gpt-4o",
    ]
    client = SimpleNamespace(models=SimpleNamespace(
        list=lambda: [SimpleNamespace(id=name) for name in names],
    ))
    assert llm.fetch_available_models(client) == [
        "gpt-4o", "gpt-4o-2024-08-06", "gpt-6-astra", "gpt-9-future",
    ]


def test_generation_reports_a_refusal_instead_of_a_json_error():
    response = SimpleNamespace(status="completed", output_text="", output=[
        SimpleNamespace(type="message", content=[
            SimpleNamespace(type="refusal", refusal="I cannot generate these questions.")
        ])
    ])
    with pytest.raises(RuntimeError, match="model refused.*cannot generate"):
        generate(response)


def test_generation_reports_the_incomplete_reason():
    response = SimpleNamespace(
        status="incomplete", output_text='{"questions": [',
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    with pytest.raises(RuntimeError, match="incomplete.*max_output_tokens"):
        generate(response)


def test_generation_reports_a_response_error():
    response = SimpleNamespace(
        status="failed", output_text="",
        error=SimpleNamespace(code="server_error", message="Please retry this request."),
    )
    with pytest.raises(RuntimeError, match="server_error.*Please retry"):
        generate(response)


@pytest.mark.parametrize("status", ["cancelled", "queued", "in_progress"])
def test_generation_does_not_accept_unfinished_response_text(status):
    response = SimpleNamespace(status=status, output_text='{"questions": []}')
    with pytest.raises(RuntimeError, match=status):
        generate(response)


def test_generation_explains_an_empty_response():
    with pytest.raises(RuntimeError, match="no question data"):
        generate(SimpleNamespace(output_text=""))


@pytest.mark.parametrize("payload", [None, [], {}, {"questions": None}, {"questions": {}}])
def test_generation_rejects_an_invalid_top_level_shape(payload):
    with pytest.raises(ValueError, match="questions array"):
        generate(SimpleNamespace(output_text=json.dumps(payload)))


@pytest.mark.parametrize("bad_answer", [None, "", "E", "AB", 1, ["A"]])
def test_bad_answer_does_not_default_to_a_or_discard_valid_questions(bad_answer):
    valid = {
        "question": "What is correct?", "option_a": "First", "option_b": "Second",
        "option_c": "Third", "option_d": "Fourth", "answer": "B",
    }
    bad = dict(valid, answer=bad_answer)
    questions = generate(SimpleNamespace(output_text=json.dumps({"questions": [bad, valid]})))
    good, problems = quiz.validate(questions)
    assert len(good) == 1
    assert good[0].answer_index == 1
    assert problems == ["Question 1: no correct answer identified - skipped."]


def test_malformed_items_preserve_the_good_questions_and_validation_notes():
    valid = {
        "question": "What is correct?", "option_a": "First", "option_b": "Second",
        "option_c": "Third", "option_d": "Fourth", "answer": "B",
    }
    payload = {"questions": [None, dict(valid, option_a={"text": "First"}), valid]}
    good, problems = quiz.validate(generate(SimpleNamespace(output_text=json.dumps(payload))))
    assert len(good) == 1
    assert good[0].answer_index == 1
    assert len(problems) == 2
