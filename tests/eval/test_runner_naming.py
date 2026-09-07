"""Run naming and the re-runnable command.

Eleven benchmark cards accumulated in `outputs/` before `bench/2`, every one of
them in a directory called `run_<stamp>_bench`, distinguishable only by opening
the JSON and reading the arm names. `--name` fixes that at both ends -- it goes
on the card verbatim and it names the directory -- so the two things worth
testing are the slug (a directory name a human can read at a glance) and
`rerun` (a command a human can actually paste).

`run.cmd` has never been pasteable: it is `" ".join(sys.argv)`, whose argv[0] is
the absolute path of `__main__.py`, and whose arguments lose their quoting. It
is kept verbatim on the card for the eyeball habit, and `rerun` is the string
that makes the claim. `test_rerun_round_trips_through_the_real_parser` is what
turns that claim into something checked: the recorded command is re-parsed
through the parser that produced the run, and must yield the same options.

That round trip is the entire reason `build_parser()` is factored out of
`main()` -- a parser constructed inside `main` cannot be reached to re-parse
anything. If this file ever gets deleted, so does the reason for the factoring.

Fast and unmarked: no corpus, no GPU, no renderer, no run directory is created
(`new_run_dir` is deliberately not called -- it writes into `outputs/`).
"""

from __future__ import annotations

import shlex

import pytest

from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench.runner import build_parser, run_slug

#: How `main` builds `rerun`: `shlex.join(["python", "-m", "<module>", *argv])`.
#: Three tokens of prefix, which is where the `[3:]` in the round trip comes
#: from -- named here so the slice is not a magic number in four assertions.
RERUN_PREFIX = ["python", "-m", "blockgen.eval.bench"]


def rerun_of(argv):
    """The exact expression `runner.main` uses. Duplicated on purpose.

    A test that imported the string from the runner would pass even if the
    runner stopped quoting; writing it out is what makes this an independent
    check of the same rule.
    """
    return shlex.join([*RERUN_PREFIX, *argv])


# --- the directory slug -----------------------------------------------------
def test_run_slug_truncates_on_a_word_boundary():
    """D43: a hard `[:40]` reads as a typo; cutting back a word reads as a name.

    `outputs/run_20260906_190029_bench_ontology_arms_vs_agentic_ontol` is worse
    than no label at all. One word is the whole price of not producing it.
    """
    slug = run_slug("ontology arms vs agentic ontology probe v2")
    assert len(slug) <= 40
    assert slug == "ontology_arms_vs_agentic_ontology_probe"
    assert not slug.endswith("_")
    # Every word that survived is a whole word.
    assert all(w for w in slug.split("_"))
    naive = "".join(c if c.isalnum() or c in "-_" else "_"
                    for c in "ontology arms vs agentic ontology probe v2")[:40]
    assert slug != naive, "the point of the boundary cut is that it differs"


def test_run_slug_leaves_a_short_name_alone():
    assert run_slug("smoke run") == "smoke_run"
    assert run_slug("ontology arms v2") == "ontology_arms_v2"


def test_run_slug_of_empty_and_punctuation():
    assert run_slug("") == ""
    assert run_slug("   ") == ""
    assert run_slug("a/b:c") == "a_b_c"
    # Runs of separators collapse, and the result never starts or ends with one:
    # `run_<stamp>_bench_` with a dangling underscore is a directory that looks
    # truncated when it is not.
    assert run_slug("a // b") == "a_b"
    assert run_slug("!name!") == "name"


def test_run_slug_hard_cuts_a_word_with_no_boundary():
    """A single 49-character word has nothing to cut back to. Still better than
    a 90-character directory name."""
    slug = run_slug("supercalifragilisticexpialidociousandthensomemore")
    assert len(slug) == 40
    assert slug.startswith("supercalifragilistic")


def test_run_slug_preserves_case():
    """`utils.runs.new_run_dir` does not fold case either, so a lowercasing slug
    would produce the one directory in `outputs/` that disagrees with the name
    it was given."""
    assert run_slug("Ontology ARMS") == "Ontology_ARMS"


def test_run_slug_keeps_hyphens_and_underscores():
    assert run_slug("pick-and-place vs ar") == "pick-and-place_vs_ar"
    assert run_slug("already_slugged") == "already_slugged"


# --- the re-runnable command ------------------------------------------------
def test_rerun_round_trips_through_the_real_parser():
    """D42: the recorded command re-parses to the run that recorded it.

    Two assertions, and both are needed. The token one proves the quoting
    survives -- `--name "ontology arms v2"` and an `--out` path containing a
    space are exactly what `" ".join(sys.argv)` destroys. The namespace one
    proves the string is not merely well-formed but *accepted*: a flag renamed
    in `build_parser` without a matching change to what the runner records
    fails here rather than on the day someone pastes the command.
    """
    argv = ["--tier", "both", "--n", "128", "--min-n", "8",
            "--name", "ontology arms v2",
            "--note", "n=128, mined vs shuffled ontology, seed 0",
            "--out", "outputs/a dir with spaces",
            "--arms", "ar/native:outputs/bench_arms/native_oriented_32.npz",
            "--examples", "4"]
    rerun = rerun_of(argv)

    assert shlex.split(rerun)[:3] == RERUN_PREFIX
    assert shlex.split(rerun)[3:] == argv, "quoting did not survive the join"
    assert build_parser().parse_args(argv) == \
        build_parser().parse_args(shlex.split(rerun)[3:])


def test_rerun_of_an_empty_argv_is_still_a_valid_command():
    """A bare `python -m blockgen.eval.bench` is the default run, and the card
    for it must not record something unpasteable."""
    rerun = rerun_of([])
    assert rerun == "python -m blockgen.eval.bench"
    args = build_parser().parse_args(shlex.split(rerun)[3:])
    assert args.tier == "fast" and args.arms == []


@pytest.mark.parametrize("name", ["ontology arms v2", "it's a run",
                                  'quotes "and" spaces', "a;b && c", ""])
def test_rerun_survives_hostile_names(name):
    """`--name` is free text, verbatim, and reaches a shell only through this
    string. Whatever a user types must come back out the same and must not be
    able to become a second command."""
    argv = ["--name", name]
    parsed = build_parser().parse_args(shlex.split(rerun_of(argv))[3:])
    assert parsed.name == name


def test_cmd_style_join_is_the_thing_that_does_not_round_trip():
    """The negative control for `rerun`, and the reason `cmd` is marked legacy.

    Naive `" ".join(argv)` re-splits into different tokens the moment an
    argument contains a space -- which is the normal case for `--name`.
    """
    argv = ["--name", "ontology arms v2"]
    assert shlex.split(" ".join(argv)) != argv
    assert shlex.split(rerun_of(argv))[3:] == argv


# --- parser defaults --------------------------------------------------------
def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.name == ""              # unnamed is the empty string, never None
    assert args.note == ""
    assert args.examples == 8
    assert args.examples == fast_tier.EXAMPLES_K, (
        "the --examples default and the writer's own K must be one number")
    assert args.tier == "fast"
    assert args.out is None
    assert args.corpus == "houses_32"


def test_examples_zero_is_expressible():
    """`--examples 0` disables the sibling artifact; it must parse as 0 and not
    fall back to the default."""
    assert build_parser().parse_args(["--examples", "0"]).examples == 0


def test_name_is_taken_verbatim_including_spaces():
    """`run.name` is free text and is never an id: no sanitising happens at
    parse time, and `run_slug` is applied only to the directory."""
    args = build_parser().parse_args(["--name", "ontology arms v2"])
    assert args.name == "ontology arms v2"
    assert run_slug(args.name) == "ontology_arms_v2"
