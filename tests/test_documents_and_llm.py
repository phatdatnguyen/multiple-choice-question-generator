import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import documents  # noqa: E402
import llm  # noqa: E402

MODEL = "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def test_extract_text_reads_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")
    assert documents.extract_text(str(path)) == "hello world"


def test_extract_text_handles_bom_and_cp1252(tmp_path):
    utf8 = tmp_path / "bom.txt"
    utf8.write_bytes("﻿with bom".encode("utf-8"))
    assert documents.extract_text(str(utf8)) == "with bom"

    legacy = tmp_path / "legacy.txt"
    legacy.write_bytes("caf\xe9 na\xefve".encode("cp1252"))
    assert "caf" in documents.extract_text(str(legacy))


def test_extract_text_rejects_empty_file(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(documents.DocumentError, match="No text found"):
        documents.extract_text(str(path))


def test_extract_text_rejects_legacy_office_formats(tmp_path):
    path = tmp_path / "old.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(documents.DocumentError, match="Word 97-2003"):
        documents.extract_text(str(path))


def test_extract_text_reports_unreadable_file(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not really a docx")
    with pytest.raises(documents.DocumentError, match="Could not read"):
        documents.extract_text(str(path))


def test_extract_text_caches_by_content(tmp_path):
    path = tmp_path / "cached.txt"
    path.write_text("original", encoding="utf-8")
    assert documents.extract_text(str(path)) == "original"

    # Same content, different file: served from cache without re-reading.
    copy = tmp_path / "copy.txt"
    copy.write_text("original", encoding="utf-8")
    assert documents.extract_text(str(copy)) == "original"

    # Changed content must produce a fresh extraction, not the cached value.
    path.write_text("changed", encoding="utf-8")
    assert documents.extract_text(str(path)) == "changed"


def test_extract_text_reads_html(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><body><h1>Title</h1><p>Body text</p></body></html>", encoding="utf-8")
    text = documents.extract_text(str(path))
    assert "Title" in text and "Body text" in text


def test_extract_text_reads_epub_in_spine_order(tmp_path):
    path = tmp_path / "book.epub"
    opf = """<?xml version="1.0"?>
    <package xmlns="http://www.idpf.org/2007/opf">
      <manifest>
        <item id="two" href="second.xhtml" media-type="application/xhtml+xml"/>
        <item id="one" href="first.xhtml" media-type="application/xhtml+xml"/>
      </manifest>
      <spine><itemref idref="one"/><itemref idref="two"/></spine>
    </package>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.opf", opf)
        archive.writestr("first.xhtml", "<html><body><p>Chapter one</p></body></html>")
        archive.writestr("second.xhtml", "<html><body><p>Chapter two</p></body></html>")

    text = documents.extract_text(str(path))
    assert text.index("Chapter one") < text.index("Chapter two")


def test_build_document_context_labels_files_and_reports_failures(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("readable content", encoding="utf-8")
    bad = tmp_path / "bad.doc"
    bad.write_bytes(b"\xd0\xcf")

    text, problems = documents.build_document_context([str(good), str(bad)])
    assert "File: good.txt" in text
    assert "readable content" in text
    assert len(problems) == 1


def test_build_document_context_accepts_a_single_path(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("solo", encoding="utf-8")
    text, problems = documents.build_document_context(str(path))
    assert "solo" in text and problems == []


def test_build_document_context_handles_no_files():
    assert documents.build_document_context(None) == ("", [])


# --------------------------------------------------------------------------- #
# Tokens and chunking
# --------------------------------------------------------------------------- #

def test_count_tokens_is_positive_and_grows():
    assert documents.count_tokens("", MODEL) == 0
    assert documents.count_tokens("hello", MODEL) >= 1
    assert documents.count_tokens("hello " * 50, MODEL) > documents.count_tokens("hello", MODEL)


def test_count_tokens_falls_back_for_unknown_model():
    assert documents.count_tokens("hello world", "some-future-model-9000") >= 2


def test_chunk_text_returns_single_chunk_when_it_fits():
    assert documents.chunk_text("short text", 1000, MODEL) == ["short text"]


def test_chunk_text_respects_the_token_limit():
    text = "\n\n".join(f"Paragraph number {i} with a little filler text." for i in range(80))
    chunks = documents.chunk_text(text, 60, MODEL)
    assert len(chunks) > 1
    assert all(documents.count_tokens(chunk, MODEL) <= 60 for chunk in chunks)


def test_chunk_text_splits_an_oversized_single_line():
    chunks = documents.chunk_text("word " * 400, 50, MODEL)
    assert len(chunks) > 1
    assert all(documents.count_tokens(chunk, MODEL) <= 50 for chunk in chunks)


def test_chunk_text_keeps_all_the_words():
    text = "\n\n".join(f"unique-token-{i}" for i in range(40))
    joined = " ".join(documents.chunk_text(text, 40, MODEL))
    assert all(f"unique-token-{i}" in joined for i in range(40))


def test_chunk_text_rejects_a_zero_budget():
    with pytest.raises(ValueError):
        documents.chunk_text("text", 0, MODEL)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model", ["o1", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.6-luna"])
def test_reasoning_models_are_recognised(model):
    assert llm.is_reasoning_model(model)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo", "gpt-5.3-chat"])
def test_chat_models_are_not_reasoning_models(model):
    assert not llm.is_reasoning_model(model)


def test_reasoning_models_get_no_sampling_parameters():
    """o-series and gpt-5 reject temperature/top_p outright."""
    kwargs = llm._request_kwargs("o3")
    assert "temperature" not in kwargs and "top_p" not in kwargs


def test_chat_models_get_sampling_parameters():
    kwargs = llm._request_kwargs("gpt-4o")
    assert kwargs["temperature"] == 1 and kwargs["top_p"] == 1


def test_deep_research_models_get_a_tool():
    """These models refuse a request with no tools attached."""
    assert llm._request_kwargs("o3-deep-research")["tools"]


def test_unknown_model_falls_back_without_crashing():
    info = llm.get_model_info("gpt-9-experimental")
    assert info.context_tokens > 0 and info.reasoning is True


def test_every_offered_model_is_in_the_registry():
    """The dropdown is generated from the registry, so the two cannot drift."""
    assert llm.MODEL_CHOICES == list(llm.MODEL_REGISTRY)
    assert llm.DEFAULT_MODEL in llm.MODEL_REGISTRY


def test_document_token_budget_is_smaller_than_the_context_window():
    for model in ("gpt-4", "gpt-4o", "o3", "gpt-5.6-luna"):
        budget = llm.document_token_budget(model, 20)
        assert 0 < budget < llm.get_max_context_tokens(model)


def test_split_count_distributes_evenly_and_totals_correctly():
    assert llm.split_count(10, 3) == [4, 3, 3]
    assert sum(llm.split_count(97, 7)) == 97
    assert llm.split_count(2, 5) == [1, 1, 0, 0, 0]
    assert llm.split_count(5, 0) == []


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

def test_build_prompt_includes_the_requested_count():
    assert "exactly 25" in llm.build_prompt("English", 25)


def test_vietnamese_prompt_keeps_the_count_and_is_vietnamese():
    """Regression: the Vietnamese branch used to discard the count entirely."""
    prompt = llm.build_prompt("Vietnamese", 25)
    assert "25" in prompt
    assert "câu hỏi trắc nghiệm" in prompt


def test_build_prompt_appends_extra_instructions():
    prompt = llm.build_prompt("English", 5, "  focus on chapter 3  ")
    assert "Additional instructions:" in prompt
    assert "focus on chapter 3" in prompt


def test_build_prompt_omits_empty_extra_instructions():
    assert "Additional instructions" not in llm.build_prompt("English", 5, "   ")


def test_unknown_language_falls_back_to_english():
    assert "exactly 5" in llm.build_prompt("Klingon", 5)


def test_question_schema_is_strict():
    schema = llm.QUESTION_SCHEMA
    item = schema["properties"]["questions"]["items"]
    assert schema["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert item["properties"]["answer"]["enum"] == ["A", "B", "C", "D"]
    assert set(item["required"]) == set(item["properties"])


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #

def test_load_dotenv_sets_missing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="sk-from-dotenv"\n# comment\nOTHER=value\n', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm.load_dotenv(str(env))
    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"


def test_load_dotenv_does_not_override_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    llm.load_dotenv(str(env))
    assert os.environ["OPENAI_API_KEY"] == "sk-from-env"


def test_load_dotenv_ignores_a_missing_file(tmp_path):
    llm.load_dotenv(str(tmp_path / "nope.env"))  # must not raise


def test_resolve_api_key_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-wins")
    assert llm.resolve_api_key() == "sk-env-wins"


# --------------------------------------------------------------------------- #
# The httpx2 + Brotli workaround
# --------------------------------------------------------------------------- #

class FakeBrotliModule:
    def __init__(self, decompressor):
        self._decompressor = decompressor

    def Decompressor(self):
        return self._decompressor


class PositionalOnlyDecompressor:
    """Mimics the `brotli` package, whose process() rejects keywords."""

    def process(self, data, *args, **kwargs):
        if kwargs:
            raise TypeError("process() takes no keyword arguments")
        return b""


class WorkingDecompressor:
    """Mimics `brotlicffi`, which httpx2 calls through a different branch."""

    def decompress(self, data, **kwargs):
        return b""


class FixedDecompressor:
    def process(self, data, output_buffer_limit=None):
        return b""


def _fake_brotli(monkeypatch, decompressor):
    monkeypatch.setitem(sys.modules, "brotli", FakeBrotliModule(decompressor))


def test_brotli_detection_flags_the_positional_only_package(monkeypatch):
    _fake_brotli(monkeypatch, PositionalOnlyDecompressor())
    assert llm.brotli_decoding_is_broken() is True


def test_brotli_detection_accepts_brotlicffi(monkeypatch):
    _fake_brotli(monkeypatch, WorkingDecompressor())
    assert llm.brotli_decoding_is_broken() is False


def test_brotli_detection_accepts_a_fixed_package(monkeypatch):
    _fake_brotli(monkeypatch, FixedDecompressor())
    assert llm.brotli_decoding_is_broken() is False


def test_brotli_detection_handles_brotli_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "brotli", None)  # forces ImportError
    assert llm.brotli_decoding_is_broken() is False


def test_client_asks_for_gzip_when_brotli_decoding_is_broken(monkeypatch):
    """Otherwise every response fails as an opaque "Connection error"."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "brotli_decoding_is_broken", lambda: True)
    encoding = llm.build_client()._client.headers.get("accept-encoding", "")
    assert "br" not in encoding.split(", ")
    assert "gzip" in encoding


def test_client_keeps_the_sdk_timeout_when_overriding_the_transport():
    """A bare httpx client would default to 5s and kill long generations."""
    client = llm.build_client()
    assert client._client.timeout.read >= 600
    assert client._client.follow_redirects is True


def test_client_leaves_the_transport_alone_when_brotli_is_healthy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "brotli_decoding_is_broken", lambda: False)
    assert "br" in llm.build_client()._client.headers.get("accept-encoding", "")


# --------------------------------------------------------------------------- #
# Error reporting
# --------------------------------------------------------------------------- #

def test_describe_error_reveals_the_underlying_cause():
    """The SDK hides real failures behind a bare "Connection error"."""
    try:
        try:
            raise TypeError("process() takes no keyword arguments")
        except TypeError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as exc:
        description = llm.describe_error(exc)

    assert "Connection error." in description
    assert "process() takes no keyword arguments" in description
    assert description.index("Connection") < description.index("process()")


def test_describe_error_handles_a_lone_exception():
    assert llm.describe_error(ValueError("bad input")) == "ValueError: bad input"


def test_describe_error_handles_an_empty_message():
    assert llm.describe_error(RuntimeError()) == "RuntimeError"


def test_describe_error_stops_on_a_cyclic_cause():
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    assert llm.describe_error(first).count("first") == 1
