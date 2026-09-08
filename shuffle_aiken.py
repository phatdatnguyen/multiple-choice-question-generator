"""Shuffle, sample and convert quiz files from the command line.

Reads Aiken format (question, options, ``ANSWER:`` line). Files with no ANSWER
lines are handled with ``--ignore-answers``.
"""

import argparse
import random
import sys

import quiz

# Short names for the command line; the UI uses the descriptive labels.
FORMAT_ALIASES = {
    "aiken": "Aiken (.txt)",
    "gift": "GIFT (.txt)",
    "xml": "Moodle XML (.xml)",
    "csv": "CSV (.csv)",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="shuffle_aiken.py",
        description="Shuffle, sample and convert Aiken quiz files.",
    )
    parser.add_argument("input", help="input file")
    parser.add_argument("output", help="output file")
    parser.add_argument(
        "-n", "--n-questions", "--n_questions", type=int, default=None, dest="n_questions",
        help="keep only this many questions, chosen at random",
    )
    parser.add_argument(
        "--shuffle-answers", action="store_true",
        help="shuffle the options within each question",
    )
    parser.add_argument(
        "--no-shuffle-questions", action="store_true",
        help="keep the original question order",
    )
    parser.add_argument(
        "--ignore-answers", action="store_true",
        help="input has no ANSWER lines (replaces the old shuffle_aiken_no_answer.py)",
    )
    parser.add_argument(
        "--format", default="aiken", choices=sorted(FORMAT_ALIASES),
        help="output format (default: aiken)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible shuffles")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    require_answers = not args.ignore_answers
    export_format = FORMAT_ALIASES[args.format]

    try:
        with open(args.input, "r", encoding="utf-8-sig") as file:
            text = file.read()
    except (OSError, UnicodeError) as exc:
        print(f"Error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 1

    questions, problems = quiz.parse_aiken(text, require_answers=require_answers)
    questions, validation_problems = quiz.validate(questions, require_answers=require_answers)
    problems.extend(validation_problems)
    for problem in problems:
        print(f"Warning: {problem}", file=sys.stderr)

    if not questions:
        print("Error: no questions found in the input file.", file=sys.stderr)
        return 1

    if args.ignore_answers and args.format != "aiken":
        print(
            f"Error: {export_format} needs a correct answer for each question; "
            f"use --format aiken with --ignore-answers.",
            file=sys.stderr,
        )
        return 1

    try:
        result = quiz.shuffle_quiz(
            questions,
            shuffle_order=not args.no_shuffle_questions,
            shuffle_answers=args.shuffle_answers,
            sample=args.n_questions,
            rng=rng,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    text_out, _ = quiz.serialise(result, export_format)
    try:
        with open(args.output, "w", encoding="utf-8", newline="\n") as file:
            file.write(text_out)
    except OSError as exc:
        print(f"Error: cannot write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(result)} questions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
