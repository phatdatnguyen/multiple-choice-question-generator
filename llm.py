"""OpenAI client setup, the model registry, and question generation."""

import json
import os
import re

from openai import DefaultHttpxClient, OpenAI

from quiz import Question

DEFAULT_MODEL = "gpt-5.6-luna"

# Reserve room for the prompt and the model's own output when deciding how much
# document text can go into a single request.
OUTPUT_TOKEN_RESERVE = 16000
PROMPT_TOKEN_RESERVE = 2000

# Tokens roughly needed per generated question, used to size the reserve.
TOKENS_PER_QUESTION = 220


class ModelInfo:
    """What we need to know about a model to call it correctly."""

    def __init__(self, context_tokens, *, reasoning=False, needs_tools=False):
        self.context_tokens = context_tokens
        self.reasoning = reasoning
        self.needs_tools = needs_tools


# Context windows in tokens. Reasoning models reject temperature/top_p, and the
# deep-research models require at least one tool to be attached.
MODEL_REGISTRY = {
    "gpt-3.5-turbo": ModelInfo(16385),
    "gpt-4": ModelInfo(8192),
    "gpt-4-turbo": ModelInfo(128000),
    "gpt-4.1": ModelInfo(1047576),
    "gpt-4.1-mini": ModelInfo(1047576),
    "gpt-4.1-nano": ModelInfo(1047576),
    "gpt-4o": ModelInfo(128000),
    "gpt-4o-mini": ModelInfo(128000),
    "gpt-5": ModelInfo(400000, reasoning=True),
    "gpt-5-mini": ModelInfo(400000, reasoning=True),
    "gpt-5-nano": ModelInfo(400000, reasoning=True),
    "gpt-5-pro": ModelInfo(400000, reasoning=True),
    "gpt-5-chat-latest": ModelInfo(400000),
    "gpt-5.1": ModelInfo(400000, reasoning=True),
    "gpt-5.1-chat-latest": ModelInfo(400000),
    "gpt-5.2": ModelInfo(400000, reasoning=True),
    "gpt-5.2-pro": ModelInfo(400000, reasoning=True),
    "gpt-5.2-chat-latest": ModelInfo(400000),
    "gpt-5.3-chat-latest": ModelInfo(128000),
    "gpt-5.4": ModelInfo(1050000, reasoning=True),
    "gpt-5.4-mini": ModelInfo(1050000, reasoning=True),
    "gpt-5.4-nano": ModelInfo(1050000, reasoning=True),
    "gpt-5.4-pro": ModelInfo(1050000, reasoning=True),
    "gpt-5.5": ModelInfo(1050000, reasoning=True),
    "gpt-5.5-pro": ModelInfo(1050000, reasoning=True),
    "gpt-5.6-luna": ModelInfo(1050000, reasoning=True),
    "gpt-5.6-terra": ModelInfo(1050000, reasoning=True),
    "gpt-5.6-sol": ModelInfo(1050000, reasoning=True),
    "o1": ModelInfo(200000, reasoning=True),
    "o1-pro": ModelInfo(200000, reasoning=True),
    "o3": ModelInfo(200000, reasoning=True),
    "o3-mini": ModelInfo(200000, reasoning=True),
    "o3-pro": ModelInfo(200000, reasoning=True),
    "o3-deep-research": ModelInfo(200000, reasoning=True, needs_tools=True),
    "o4-mini": ModelInfo(200000, reasoning=True),
    "o4-mini-deep-research": ModelInfo(200000, reasoning=True, needs_tools=True),
}

MODEL_CHOICES = list(MODEL_REGISTRY)

_FALLBACK_CONTEXT_TOKENS = 128000


def get_model_info(model_name):
    """Look up a model, guessing conservatively for IDs we do not know about.

    An unknown model is assumed to be a reasoning model, so no temperature or
    top_p is sent. Both values would be 1 - the API default - so omitting them
    changes nothing for a chat model, whereas sending them to a reasoning model
    is a hard 400.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    name = str(model_name or "").lower()
    return ModelInfo(
        _FALLBACK_CONTEXT_TOKENS,
        reasoning="chat" not in name,
        needs_tools="deep-research" in name,
    )


def get_max_context_tokens(model_name):
    return get_model_info(model_name).context_tokens


def is_reasoning_model(model_name):
    return get_model_info(model_name).reasoning


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #

class MissingAPIKey(RuntimeError):
    pass


def describe_error(exc):
    """Describe an exception including its underlying cause.

    The SDK reports several unrelated low-level failures as a bare "Connection
    error", which leaves nothing to act on, so the whole cause chain is shown.
    """
    parts, seen, current = [], set(), exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        detail = str(current).strip()
        parts.append(f"{type(current).__name__}: {detail}" if detail else type(current).__name__)
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def load_dotenv(path=".env"):
    """Load ``KEY=value`` pairs from a .env file into the environment.

    Existing environment variables win, and no third-party package is needed.
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_api_key():
    """Find the API key: environment first, then .env, then api_key.py."""
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    try:  # Legacy location, kept working for existing checkouts.
        from api_key import API_KEY
    except ImportError:
        API_KEY = ""
    if str(API_KEY or "").strip():
        return str(API_KEY).strip()

    raise MissingAPIKey(
        "No OpenAI API key found. Set the OPENAI_API_KEY environment variable, "
        "or create a .env file next to webui.py containing "
        "OPENAI_API_KEY=sk-your-key-here"
    )


def brotli_decoding_is_broken():
    """Detect the httpx2 + Brotli incompatibility described in build_client()."""
    try:
        import brotli
    except ImportError:
        return False

    decompressor = brotli.Decompressor()
    if hasattr(decompressor, "decompress"):
        return False  # brotlicffi, which httpx2 calls correctly

    try:
        decompressor.process(b"", output_buffer_limit=1)
    except TypeError:
        return True
    except Exception:
        pass
    return False


def build_client():
    """Create the OpenAI client.

    httpx2 2.12.0 decodes brotli responses by calling the decompressor with an
    ``output_buffer_limit`` keyword. The ``brotli`` package's ``process()`` takes
    a single positional argument, so every brotli-encoded response raises a
    TypeError that the SDK reports only as "Connection error". httpx2 imports
    ``brotli`` in preference to ``brotlicffi`` and gradio always installs it, so
    the fix is to stop advertising brotli support. gzip is used instead, which
    costs a little bandwidth and nothing else.
    """
    api_key = resolve_api_key()
    if not brotli_decoding_is_broken():
        return OpenAI(api_key=api_key)
    # DefaultHttpxClient keeps the SDK's own timeout, limits and redirect settings.
    return OpenAI(
        api_key=api_key,
        http_client=DefaultHttpxClient(headers={"Accept-Encoding": "gzip, deflate"}),
    )


def fetch_available_models(client, *, keep_prefixes=("gpt-", "o1", "o3", "o4", "chatgpt-")):
    """List model IDs the account can actually use.

    Lets you confirm a hardcoded ID still exists instead of discovering it via a
    404 mid-generation.
    """
    names = [model.id for model in client.models.list()]
    filtered = [
        name for name in names
        if name.startswith(keep_prefixes)
        and not re.search(r"(audio|realtime|transcribe|tts|image|embedding|moderation|search-preview)", name)
    ]
    return sorted(set(filtered))


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

BASE_PROMPTS = {
    "English": (
        "Generate exactly {count} multiple-choice questions based on the content of the "
        "following document. Write the questions and options in English. Each question "
        "must have exactly four plausible options and exactly one correct answer. Base "
        "every question strictly on the document; do not invent facts. Do not repeat or "
        "rephrase the same question."
    ),
    "Vietnamese": (
        "Tạo chính xác {count} câu hỏi trắc nghiệm dựa trên nội dung của tài liệu sau. "
        "Viết câu hỏi và các lựa chọn bằng tiếng Việt. Mỗi câu hỏi phải có đúng bốn lựa "
        "chọn hợp lý và chỉ một đáp án đúng. Chỉ dựa vào nội dung tài liệu; không tự "
        "thêm thông tin. Không lặp lại hoặc diễn đạt lại cùng một câu hỏi."
    ),
}


def build_prompt(language, count, extra_instructions=""):
    """Assemble the instruction sent to the model."""
    template = BASE_PROMPTS.get(language, BASE_PROMPTS["English"])
    prompt = template.format(count=count)
    extra = str(extra_instructions or "").strip()
    if extra:
        prompt = f"{prompt}\n\nAdditional instructions:\n{extra}"
    return prompt


# The response shape is enforced by the API rather than parsed out of prose, so
# fenced code blocks and stray commentary can no longer break generation.
# Strict mode does not support minItems/maxItems, hence four explicit fields.
QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "option_a", "option_b", "option_c", "option_d", "answer"],
                "properties": {
                    "question": {"type": "string", "description": "The question stem, on a single line."},
                    "option_a": {"type": "string"},
                    "option_b": {"type": "string"},
                    "option_c": {"type": "string"},
                    "option_d": {"type": "string"},
                    "answer": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Letter of the correct option.",
                    },
                },
            },
        }
    },
}


def _request_kwargs(model_name):
    """Build model-specific request arguments.

    Reasoning models reject temperature/top_p, and deep-research models refuse a
    request with no tools attached.
    """
    info = get_model_info(model_name)
    kwargs = {}
    if not info.reasoning:
        kwargs["temperature"] = 1
        kwargs["top_p"] = 1
    if info.needs_tools:
        kwargs["tools"] = [{"type": "web_search_preview"}]
    return kwargs


def document_token_budget(model_name, count):
    """How much document text fits in one request for this model."""
    reserve = PROMPT_TOKEN_RESERVE + max(OUTPUT_TOKEN_RESERVE, count * TOKENS_PER_QUESTION)
    if is_reasoning_model(model_name):
        reserve += OUTPUT_TOKEN_RESERVE  # reasoning tokens also count as output
    return max(2000, get_max_context_tokens(model_name) - reserve)


def split_count(total, parts):
    """Split ``total`` questions as evenly as possible across ``parts`` chunks."""
    if parts <= 0:
        return []
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def generate_questions(client, model_name, prompt, document_text):
    """Ask the model for questions and return validated ``Question`` objects."""
    response = client.responses.create(
        model=model_name,
        instructions=prompt,
        input=document_text,
        text={
            "format": {
                "type": "json_schema",
                "name": "multiple_choice_questions",
                "strict": True,
                "schema": QUESTION_SCHEMA,
            }
        },
        **_request_kwargs(model_name),
    )

    payload = json.loads(response.output_text)
    questions = []
    for item in payload.get("questions", []):
        options = [item.get(f"option_{letter}", "") for letter in ("a", "b", "c", "d")]
        answer_index = "ABCD".index(item.get("answer", "A"))
        questions.append(Question(text=item.get("question", ""), options=options, answer_index=answer_index))
    return questions
