import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quiz  # noqa: E402


AIKEN_SAMPLE = """How many natural amino acids are present in the human body?
A. 20
B. 21
C. 22
D. 24
ANSWER: C

Which pair is identified as the 2 specialized natural amino acids?
A. Hydroxyproline and ornithine
B. Selenocysteine and pyrrolysine
C. Cystine and tyrosine
D. Arginine and histidine
ANSWER: B
"""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_parse_aiken_reads_all_questions():
    questions, problems = quiz.parse_aiken(AIKEN_SAMPLE)
    assert problems == []
    assert len(questions) == 2
    assert questions[0].answer_label == "C"
    assert questions[0].options[2] == "22"
    assert questions[1].answer_label == "B"


def test_parse_aiken_accepts_paren_and_colon_labels():
    text = "Pick one\nA) first\nB) second\nANSWER: B"
    questions, problems = quiz.parse_aiken(text)
    assert problems == []
    assert questions[0].options == ["first", "second"]
    assert questions[0].answer_label == "B"


def test_parse_aiken_joins_multi_line_stem():
    text = "This stem spans\ntwo lines\nA. yes\nB. no\nANSWER: A"
    questions, _ = quiz.parse_aiken(text)
    assert questions[0].text == "This stem spans two lines"


def test_parse_aiken_joins_wrapped_option_text():
    """A wrapped option must stay with its option, not jump into the stem."""
    text = "Question?\nA. a very long option that\ncontinues on the next line\nB. short\nANSWER: A"
    questions, _ = quiz.parse_aiken(text)
    assert questions[0].text == "Question?"
    assert questions[0].options == ["a very long option that continues on the next line", "short"]


def test_parse_aiken_does_not_treat_initials_in_stem_as_options():
    """A stem beginning 'A. ' is a stem, because options follow a stem."""
    text = "A. Einstein proposed which theory?\nA. Relativity\nB. Gravity\nANSWER: A"
    questions, _ = quiz.parse_aiken(text)
    assert questions[0].text == "A. Einstein proposed which theory?"
    assert questions[0].options == ["Relativity", "Gravity"]


def test_parse_aiken_reports_question_missing_answer():
    text = "First?\nA. one\nB. two\nANSWER: A\n\nSecond?\nA. one\nB. two\n"
    questions, problems = quiz.parse_aiken(text)
    assert len(questions) == 1
    assert len(problems) == 1
    assert "no ANSWER line" in problems[0]


def test_parse_aiken_recovers_after_question_missing_answer():
    text = "Broken?\nA. one\nB. two\n\nGood?\nA. three\nB. four\nANSWER: B"
    questions, problems = quiz.parse_aiken(text)
    assert len(problems) == 1
    assert "no ANSWER line" in problems[0]
    assert questions == [quiz.Question("Good?", ["three", "four"], 1)]


def test_parse_aiken_allows_blank_lines_before_options_and_answer():
    text = "Q?\nA. one\n\nB. two\n\nANSWER: B"
    questions, problems = quiz.parse_aiken(text)
    assert problems == []
    assert questions == [quiz.Question("Q?", ["one", "two"], 1)]


def test_parse_aiken_reports_unrecognised_answer():
    questions, problems = quiz.parse_aiken("Q?\nA. one\nB. two\nANSWER: Z")
    assert questions == []
    assert "unrecognised answer" in problems[0]


def test_parse_aiken_without_answers():
    text = "First?\nA. one\nB. two\n\nSecond?\nA. three\nB. four\n"
    questions, problems = quiz.parse_aiken(text, require_answers=False)
    assert problems == []
    assert len(questions) == 2
    assert questions[1].options == ["three", "four"]
    assert questions[1].answer_index is None


def test_parse_aiken_accepts_vietnamese_answer_keyword():
    questions, problems = quiz.parse_aiken("Câu hỏi?\nA. một\nB. hai\nĐÁP ÁN: B")
    assert problems == []
    assert questions[0].answer_label == "B"


def test_parse_aiken_round_trips():
    questions, _ = quiz.parse_aiken(AIKEN_SAMPLE)
    reparsed, problems = quiz.parse_aiken(quiz.to_aiken(questions))
    assert problems == []
    assert [q.text for q in reparsed] == [q.text for q in questions]
    assert [q.answer_label for q in reparsed] == [q.answer_label for q in questions]


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("B", 1), ("b", 1), ("B.", 1), ("(B)", 1), ("Option B", 1), ("ANSWER: B", 1),
    ("2", 1), ("second", 1), ("SECOND", 1), ("", None), (None, None), ("Z", None),
])
def test_parse_answer_accepts_many_shapes(raw, expected):
    assert quiz.parse_answer(raw, ["first", "second", "third"]) == expected


def test_parse_answer_rejects_letter_beyond_option_count():
    assert quiz.parse_answer("D", ["first", "second"]) is None


# --------------------------------------------------------------------------- #
# Shuffling
# --------------------------------------------------------------------------- #

def test_shuffle_options_keeps_correct_answer_with_duplicate_option_text():
    """Regression: matching options by text mislabels the answer when two match."""
    question = quiz.Question(text="Q?", options=["same", "same", "right", "other"], answer_index=2)
    for seed in range(60):
        shuffled = quiz.shuffle_options(question, rng=random.Random(seed))
        assert shuffled.options[shuffled.answer_index] == "right"
        assert sorted(shuffled.options) == sorted(question.options)


def test_shuffle_options_preserves_answer_generally():
    question = quiz.Question(text="Q?", options=["w", "x", "y", "z"], answer_index=1)
    for seed in range(30):
        shuffled = quiz.shuffle_options(question, rng=random.Random(seed))
        assert shuffled.options[shuffled.answer_index] == "x"


def test_shuffle_options_without_answer_stays_answerless():
    question = quiz.Question(text="Q?", options=["a", "b"], answer_index=None)
    assert quiz.shuffle_options(question, rng=random.Random(0)).answer_index is None


def test_shuffle_quiz_samples_requested_count():
    questions = [quiz.Question(f"Q{i}", ["a", "b"], 0) for i in range(10)]
    assert len(quiz.shuffle_quiz(questions, sample=4, rng=random.Random(1))) == 4


def test_shuffle_quiz_sampling_preserves_order_when_shuffle_order_is_false():
    questions = [quiz.Question(f"Q{i}", ["a", "b"], 0) for i in range(10)]
    sampled = quiz.shuffle_quiz(questions, sample=5, shuffle_order=False, rng=random.Random(1))
    positions = [questions.index(question) for question in sampled]
    assert len(positions) == 5
    assert positions == sorted(positions)


def test_shuffle_quiz_rejects_oversized_sample():
    questions = [quiz.Question("Q", ["a", "b"], 0)]
    with pytest.raises(ValueError, match="only 1 available"):
        quiz.shuffle_quiz(questions, sample=5)


def test_shuffle_quiz_rejects_zero_sample():
    with pytest.raises(ValueError, match="greater than 0"):
        quiz.shuffle_quiz([quiz.Question("Q", ["a", "b"], 0)], sample=0)


# --------------------------------------------------------------------------- #
# Validation and duplicates
# --------------------------------------------------------------------------- #

def test_validate_skips_broken_questions():
    questions = [
        quiz.Question("", ["a", "b"], 0),
        quiz.Question("No options", [], 0),
        quiz.Question("Bad answer", ["a", "b"], 7),
        quiz.Question("Good", ["a", "b", "c", "d"], 1),
    ]
    good, problems = quiz.validate(questions)
    assert [q.text for q in good] == ["Good"]
    assert len(problems) == 3


def test_validate_normalises_multi_line_stem():
    good, _ = quiz.validate([quiz.Question("line one\nline two", ["a", "b", "c", "d"], 0)])
    assert good[0].text == "line one line two"


def test_validate_allows_missing_answers_when_not_required():
    good, problems = quiz.validate(
        [quiz.Question("Q", ["a", "b", "c", "d"], None)], require_answers=False
    )
    assert len(good) == 1 and problems == []


def test_find_near_duplicates_flags_reworded_question():
    questions = [
        quiz.Question("How many natural amino acids are in the human body?", ["a"] * 4, 0),
        quiz.Question("How many natural amino acids are in the body human?", ["a"] * 4, 0),
        quiz.Question("Which enzyme catalyses peptide bond formation?", ["a"] * 4, 0),
    ]
    matches = quiz.find_near_duplicates(questions)
    assert [(i, j) for i, j, _ in matches] == [(0, 1)]


def test_find_near_duplicates_ignores_distinct_questions():
    questions = [
        quiz.Question("What is the capital of France?", ["a"] * 4, 0),
        quiz.Question("Which protein transports oxygen in blood?", ["a"] * 4, 0),
    ]
    assert quiz.find_near_duplicates(questions) == []


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def test_to_aiken_puts_stem_on_one_line():
    """Moodle's Aiken importer requires a single-line stem."""
    questions = [quiz.Question("multi\nline\nstem", ["a", "b", "c", "d"], 3)]
    lines = quiz.to_aiken(questions).splitlines()
    assert lines[0] == "multi line stem"
    assert lines[1:5] == ["A. a", "B. b", "C. c", "D. d"]
    assert lines[5] == "ANSWER: D"


def test_to_gift_marks_correct_option_and_escapes_specials():
    questions = [quiz.Question("Cost = 5 {each}?", ["a~b", "plain"], 0)]
    output = quiz.to_gift(questions)
    assert r"\=" in output and r"\{" in output
    assert "=a\\~b" in output
    assert "~plain" in output


def test_to_moodle_xml_is_wellformed_and_scores_the_answer():
    from xml.etree import ElementTree

    questions = [quiz.Question("Q<1> & more", ["wrong", "right"], 1)]
    root = ElementTree.fromstring(quiz.to_moodle_xml(questions))
    answers = root.findall("./question/answer")
    assert [a.get("fraction") for a in answers] == ["0", "100"]
    assert answers[1].findtext("text") == "right"


def test_to_moodle_xml_survives_cdata_terminator_in_text():
    from xml.etree import ElementTree

    questions = [quiz.Question("ends with ]]> inside", ["a", "b"], 0)]
    root = ElementTree.fromstring(quiz.to_moodle_xml(questions))
    assert "]]>" in root.findtext("./question/questiontext/text")


def test_to_moodle_xml_preserves_literal_html_syntax():
    from xml.etree import ElementTree

    question = quiz.Question("What does <br> mean?", ["A <b> tag", "A line break"], 1)
    root = ElementTree.fromstring(quiz.to_moodle_xml([question]))
    questiontext = root.find("./question/questiontext")
    assert questiontext.get("format") == "plain_text"
    assert questiontext.findtext("text") == question.text
    answers = root.findall("./question/answer")
    assert all(answer.get("format") == "plain_text" for answer in answers)
    assert answers[0].findtext("text") == question.options[0]


def test_to_csv_has_header_and_answer_letter():
    output = quiz.to_csv([quiz.Question("Q?", ["a", "b", "c", "d"], 2)])
    header, row = output.splitlines()[:2]
    assert header == "Question,Option A,Option B,Option C,Option D,Answer"
    assert row.endswith(",C")


def test_serialise_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unknown export format"):
        quiz.serialise([], "PDF")


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    "../../escape", "..\\..\\escape", "/etc/passwd", "C:\\Windows\\system32\\x",
])
def test_safe_file_stem_strips_path_components(raw):
    stem = quiz.safe_file_stem(raw)
    assert not any(part in stem for part in ("/", "\\", ".."))


def test_safe_file_stem_falls_back_for_empty_input():
    assert quiz.safe_file_stem("") == "questions"
    assert quiz.safe_file_stem("   ") == "questions"
    assert quiz.safe_file_stem("...") == "questions"


def test_safe_file_stem_keeps_ordinary_names():
    assert quiz.safe_file_stem("chapter 3_final-v2") == "chapter 3_final-v2"


def test_safe_file_stem_caps_length():
    assert len(quiz.safe_file_stem("x" * 500)) == 100


@pytest.mark.parametrize("raw", ["CON", "nul", "Aux.txt", "COM1", "lpt9.quiz", "CON .quiz"])
def test_safe_file_stem_avoids_windows_device_names(raw):
    assert quiz.safe_file_stem(raw) == "_" + raw


def test_safe_file_stem_strips_trailing_spaces_after_truncation():
    assert quiz.safe_file_stem("x" * 99 + " more") == "x" * 99


# --------------------------------------------------------------------------- #
# Dataframe round trip
# --------------------------------------------------------------------------- #

def test_from_rows_parses_and_validates():
    rows = [
        ["Q1", "a", "b", "c", "d", "B"],
        ["", "a", "b", "c", "d", "A"],
        ["Q3", "a", "b", "c", "d", "nonsense"],
    ]
    questions, problems = quiz.from_rows(rows)
    assert [q.text for q in questions] == ["Q1"]
    assert questions[0].answer_index == 1
    assert len(problems) == 2


def test_from_rows_handles_none_and_short_rows():
    questions, _ = quiz.from_rows([["Q", "a", "b", None, None, "a"]])
    assert questions[0].options == ["a", "b"]
    assert questions[0].answer_index == 0


@pytest.mark.parametrize("answer", ["C", "Option C", "3", "correct"])
def test_from_rows_keeps_answer_when_an_earlier_option_is_blank(answer):
    questions, _ = quiz.from_rows([["Q", "first", "", "correct", "last", answer]])
    assert questions == [quiz.Question("Q", ["first", "correct", "last"], 1)]


def test_from_rows_rejects_answer_pointing_to_a_cleared_option():
    questions, problems = quiz.from_rows([["Q", "first", " ", "third", "last", "B"]])
    assert questions == []
    assert "no correct answer" in problems[0]


def test_to_rows_pads_to_four_options():
    assert quiz.to_rows([quiz.Question("Q", ["a", "b"], 0)]) == [["Q", "a", "b", "", "", "A"]]


# --------------------------------------------------------------------------- #
# CLI validation
# --------------------------------------------------------------------------- #

def test_cli_skips_invalid_questions_before_export(tmp_path, capsys):
    import shuffle_aiken

    source = tmp_path / "input.txt"
    target = tmp_path / "output.txt"
    source.write_text("Invalid?\nA. one\nANSWER: A\n\n" + AIKEN_SAMPLE, encoding="utf-8")
    assert shuffle_aiken.main([str(source), str(target), "--no-shuffle-questions"]) == 0
    exported, _ = quiz.parse_aiken(target.read_text(encoding="utf-8"))
    assert len(exported) == 2
    assert "needs at least 2 non-empty options" in capsys.readouterr().err


def test_cli_reports_invalid_utf8_without_a_traceback(tmp_path, capsys):
    import shuffle_aiken

    source = tmp_path / "input.txt"
    target = tmp_path / "output.txt"
    source.write_bytes(b"\xff\xfeinvalid")
    assert shuffle_aiken.main([str(source), str(target)]) == 1
    assert "cannot read" in capsys.readouterr().err
    assert not target.exists()
