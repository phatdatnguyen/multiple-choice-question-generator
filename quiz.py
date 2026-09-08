"""Question model, quiz file formats, and shuffling.

Shared by the web UI and the command line tools so that both agree on what a
valid question looks like.
"""

import csv
import io
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from xml.sax.saxutils import escape

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Column layout used by the dataframe in the web UI.
DATAFRAME_COLUMNS = ["Question", "Option A", "Option B", "Option C", "Option D", "Answer"]

EXPORT_FORMATS = ["Aiken (.txt)", "GIFT (.txt)", "Moodle XML (.xml)", "CSV (.csv)"]

_LABEL_RE = re.compile(r"^([A-Za-z])\s*[.):]\s*(.*)$")
_ANSWER_RE = re.compile(r"^(?:ANSWER|ANS|ĐÁP\s*ÁN|DAP\s*AN)\s*[:.]?\s*(.*)$", re.IGNORECASE)
_FILENAME_RE = re.compile(r"[^A-Za-z0-9 ._-]+")
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


@dataclass
class Question:
    """A single multiple-choice question.

    The correct answer is stored as an index into ``options`` rather than as a
    letter, so relabelling after a shuffle cannot pick the wrong option when two
    options happen to have identical text.
    """

    text: str
    options: list = field(default_factory=list)
    answer_index: Optional[int] = None

    @property
    def answer_label(self):
        if self.answer_index is None:
            return ""
        return OPTION_LABELS[self.answer_index]

    @property
    def has_answer(self):
        return self.answer_index is not None


def normalise_stem(text):
    """Collapse a question stem onto a single line.

    Moodle's Aiken and GIFT importers require the stem to be one line, and
    models happily return stems containing newlines.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_answer(raw, options):
    """Resolve an answer of unknown shape to an index into ``options``.

    Accepts a bare letter, a decorated letter (``B.``, ``(B)``, ``Option B``),
    a 1-based number, or the full text of the correct option.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    stripped = re.sub(r"^(?:option|answer|đáp\s*án|dap\s*an|câu|cau)\s*", "", text, flags=re.IGNORECASE).strip()
    stripped = stripped.strip("().:[]{}<> \t")

    if len(stripped) == 1:
        upper = stripped.upper()
        if upper in OPTION_LABELS:
            index = OPTION_LABELS.index(upper)
            if index < len(options):
                return index
        if stripped.isdigit():
            index = int(stripped) - 1
            if 0 <= index < len(options):
                return index

    # Fall back to matching the option text itself.
    needle = _fold(stripped)
    if needle:
        for index, option in enumerate(options):
            if _fold(option) == needle:
                return index

    return None


def _fold(text):
    """Lowercase, strip accents and punctuation - for loose text comparison."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", without_accents.lower()).strip()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(questions, *, require_answers=True, expected_options=4):
    """Split ``questions`` into usable ones and human-readable problems."""
    good, problems = [], []
    for position, question in enumerate(questions, start=1):
        stem = normalise_stem(question.text)
        options = [normalise_stem(option) for option in question.options]

        if not stem:
            problems.append(f"Question {position}: empty question text - skipped.")
            continue
        if len(options) < 2 or any(not option for option in options):
            problems.append(f"Question {position}: needs at least 2 non-empty options - skipped.")
            continue

        answer_index = question.answer_index
        if require_answers:
            if answer_index is None:
                problems.append(f"Question {position}: no correct answer identified - skipped.")
                continue
            if not 0 <= answer_index < len(options):
                problems.append(f"Question {position}: answer is out of range - skipped.")
                continue

        # Only worth mentioning for a question we are actually keeping.
        if expected_options and len(options) != expected_options:
            problems.append(f"Question {position}: has {len(options)} options, expected {expected_options}.")

        good.append(Question(text=stem, options=options, answer_index=answer_index))
    return good, problems


def find_near_duplicates(questions, threshold=0.85):
    """Return ``(i, j, similarity)`` for question pairs that look alike.

    Uses token-set overlap on accent-folded text, which behaves sensibly for
    both English and Vietnamese without needing a language model.
    """
    token_sets = [set(_fold(question.text).split()) for question in questions]
    matches = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            left, right = token_sets[i], token_sets[j]
            if not left or not right:
                continue
            union = len(left | right)
            similarity = len(left & right) / union if union else 0.0
            if similarity >= threshold:
                matches.append((i, j, similarity))
    return matches


# --------------------------------------------------------------------------- #
# Shuffling and sampling
# --------------------------------------------------------------------------- #

def shuffle_options(question, rng=random):
    """Shuffle a question's options, keeping the correct answer correct."""
    pairs = [(option, index == question.answer_index) for index, option in enumerate(question.options)]
    rng.shuffle(pairs)
    answer_index = None
    for index, (_, is_correct) in enumerate(pairs):
        if is_correct:
            answer_index = index
    return Question(
        text=question.text,
        options=[option for option, _ in pairs],
        answer_index=answer_index,
    )


def shuffle_quiz(questions, *, shuffle_order=True, shuffle_answers=False, sample=None, rng=random):
    """Sample and shuffle a list of questions."""
    result = list(questions)
    if sample is not None:
        if sample <= 0:
            raise ValueError("Number of questions to keep must be greater than 0.")
        if sample > len(result):
            raise ValueError(f"Cannot keep {sample} questions - only {len(result)} available.")
        indices = rng.sample(range(len(result)), sample)
        if not shuffle_order:
            indices.sort()
        result = [result[index] for index in indices]
    if shuffle_order:
        rng.shuffle(result)
    if shuffle_answers:
        result = [shuffle_options(question, rng=rng) for question in result]
    return result


# --------------------------------------------------------------------------- #
# Aiken
# --------------------------------------------------------------------------- #

def parse_aiken(text, *, require_answers=True):
    """Parse Aiken-formatted text into questions.

    Returns ``(questions, problems)``. Handles multi-line stems, options wrapped
    across lines, and ``A.`` / ``A)`` / ``A:`` labels. A line is only treated as
    an option when its letter is the next one expected and a stem has already
    been seen, so a stem such as ``A. Einstein proposed...`` is not mistaken for
    an option.
    """
    questions, problems = [], []
    stem_lines, options, answer_raw = [], [], None
    started_line = 0
    line_number = 0
    after_blank = False

    def flush(line_number, *, complete):
        nonlocal stem_lines, options, answer_raw
        if not stem_lines and not options:
            return
        stem = normalise_stem(" ".join(stem_lines))
        if require_answers and not complete:
            problems.append(
                f"Question near line {started_line or line_number} "
                f"(\"{stem[:50]}...\") has no ANSWER line - skipped."
            )
        else:
            answer_index = parse_answer(answer_raw, options) if require_answers else None
            if require_answers and answer_index is None:
                problems.append(
                    f"Question near line {started_line or line_number} has an "
                    f"unrecognised answer {answer_raw!r} - skipped."
                )
            else:
                questions.append(Question(text=stem, options=options, answer_index=answer_index))
        stem_lines, options, answer_raw = [], [], None

    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            # In answerless files a blank line is the only question separator.
            if not require_answers:
                flush(line_number, complete=True)
            elif options:
                after_blank = True
            continue

        if after_blank:
            expected = OPTION_LABELS[len(options)] if len(options) < len(OPTION_LABELS) else None
            label_match = _LABEL_RE.match(line)
            continues_options = label_match and label_match.group(1).upper() == expected
            if not _ANSWER_RE.match(line) and not continues_options:
                # Recover at the next question instead of absorbing its stem
                # and options into a question whose ANSWER line is missing.
                flush(line_number, complete=False)
            after_blank = False

        if not stem_lines and not options:
            started_line = line_number

        answer_match = _ANSWER_RE.match(line)
        if answer_match and options:
            answer_raw = answer_match.group(1).strip()
            flush(line_number, complete=True)
            continue

        label_match = _LABEL_RE.match(line)
        expected = OPTION_LABELS[len(options)] if len(options) < len(OPTION_LABELS) else None
        if label_match and stem_lines and label_match.group(1).upper() == expected:
            options.append(label_match.group(2).strip())
        elif options:
            # Continuation of the option we are in the middle of.
            options[-1] = f"{options[-1]} {line}".strip()
        else:
            stem_lines.append(line)

    flush(line_number, complete=not require_answers)
    return questions, problems


def to_aiken(questions):
    lines = []
    for question in questions:
        lines.append(normalise_stem(question.text))
        for index, option in enumerate(question.options):
            lines.append(f"{OPTION_LABELS[index]}. {normalise_stem(option)}")
        if question.has_answer:
            lines.append(f"ANSWER: {question.answer_label}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# GIFT
# --------------------------------------------------------------------------- #

def _escape_gift(text):
    return re.sub(r"([~=#{}:\\])", r"\\\1", normalise_stem(text))


def to_gift(questions):
    blocks = []
    for position, question in enumerate(questions, start=1):
        lines = [f"::Q{position}:: {_escape_gift(question.text)} {{"]
        for index, option in enumerate(question.options):
            prefix = "=" if index == question.answer_index else "~"
            lines.append(f"    {prefix}{_escape_gift(option)}")
        lines.append("}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


# --------------------------------------------------------------------------- #
# Moodle XML
# --------------------------------------------------------------------------- #

def _cdata(text):
    # "]]>" would close the section early; split it across two sections.
    return "<![CDATA[" + normalise_stem(text).replace("]]>", "]]]]><![CDATA[>") + "]]>"


def to_moodle_xml(questions, *, category=None):
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<quiz>"]
    if category:
        out += [
            '  <question type="category">',
            f"    <category><text>{escape(category)}</text></category>",
            "  </question>",
        ]
    for position, question in enumerate(questions, start=1):
        out += [
            '  <question type="multichoice">',
            f"    <name><text>Question {position}</text></name>",
            f'    <questiontext format="plain_text"><text>{_cdata(question.text)}</text></questiontext>',
            "    <single>true</single>",
            "    <shuffleanswers>true</shuffleanswers>",
            "    <answernumbering>abc</answernumbering>",
        ]
        for index, option in enumerate(question.options):
            fraction = 100 if index == question.answer_index else 0
            out += [
                f'    <answer fraction="{fraction}" format="plain_text">',
                f"      <text>{_cdata(option)}</text>",
                "    </answer>",
            ]
        out.append("  </question>")
    out.append("</quiz>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

def to_csv(questions):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    width = max((len(question.options) for question in questions), default=4)
    writer.writerow(["Question"] + [f"Option {OPTION_LABELS[i]}" for i in range(width)] + ["Answer"])
    for question in questions:
        options = [normalise_stem(option) for option in question.options]
        options += [""] * (width - len(options))
        writer.writerow([normalise_stem(question.text)] + options + [question.answer_label])
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Dispatch helpers
# --------------------------------------------------------------------------- #

_SERIALISERS = {
    "Aiken (.txt)": (to_aiken, ".txt"),
    "GIFT (.txt)": (to_gift, ".txt"),
    "Moodle XML (.xml)": (to_moodle_xml, ".xml"),
    "CSV (.csv)": (to_csv, ".csv"),
}


def serialise(questions, export_format):
    """Render ``questions`` in ``export_format``, returning ``(text, suffix)``."""
    try:
        serialiser, suffix = _SERIALISERS[export_format]
    except KeyError:
        raise ValueError(f"Unknown export format: {export_format!r}") from None
    return serialiser(questions), suffix


def safe_file_stem(name, *, default="questions"):
    """Reduce user input to a bare filename with no path components.

    Prevents a value such as ``../../notes`` from escaping the output folder.
    """
    candidate = _FILENAME_RE.sub("_", str(name or "").replace("\\", "/").split("/")[-1]).strip(" .")
    # Windows reserves these names even when an extension is appended.
    if candidate.split(".", 1)[0].rstrip().upper() in _WINDOWS_RESERVED_NAMES:
        candidate = "_" + candidate
    return candidate[:100].rstrip(" .") or default


# --------------------------------------------------------------------------- #
# Dataframe conversion
# --------------------------------------------------------------------------- #

def to_rows(questions):
    """Convert questions to the row layout used by the UI dataframe."""
    rows = []
    for question in questions:
        options = list(question.options[:4]) + [""] * max(0, 4 - len(question.options))
        rows.append([question.text] + options + [question.answer_label])
    return rows


def from_rows(rows, *, require_answers=True):
    """Convert UI dataframe rows back into questions.

    Rows may have been hand-edited, so everything is re-validated.
    """
    parsed = []
    for row in rows:
        values = ["" if value is None else str(value) for value in row]
        values += [""] * max(0, 6 - len(values))
        stem, option_values, answer_raw = values[0], values[1:5], values[5]
        # Resolve labels against their original columns before removing blanks.
        # Otherwise clearing B silently makes answer C refer to the old D.
        original_answer = parse_answer(answer_raw, option_values)
        kept_indices = [index for index, option in enumerate(option_values) if option.strip()]
        options = [option_values[index] for index in kept_indices]
        answer_index = kept_indices.index(original_answer) if original_answer in kept_indices else None
        parsed.append(Question(text=stem, options=options, answer_index=answer_index))
    return validate(parsed, require_answers=require_answers)
