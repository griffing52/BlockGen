"""`prerender --runs <dir>` resolves a run directory to its dataset ids.

The id for a bench run's examples is mangled from its path
(`arm:run_..._bench_x__examples_32`), so warming one run's images used to mean
reproducing that rule by hand at the command line. `_ids_under` asks the catalog
what it already decided instead of re-deriving it, which is the property these
tests pin: match on the recorded source path, never on a second slugging rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.lab.prerender import _ids_under


@dataclass
class _D:
    id: str
    source: str


class _Catalog:
    def __init__(self, datasets):
        self._d = datasets

    def list_datasets(self):
        return self._d


def _catalog(tmp_path):
    run_a = tmp_path / "run_A_bench"
    run_b = tmp_path / "run_B_bench"
    for r in (run_a, run_b):
        r.mkdir()
        (r / "examples_32.npz").touch()
    return _Catalog([
        _D("arm:run_A_bench__examples_32", str(run_a / "examples_32.npz")),
        _D("arm:run_B_bench__examples_32", str(run_b / "examples_32.npz")),
        _D("corpus:houses_32", str(tmp_path / "houses_32.npz")),
    ]), run_a, run_b


def test_a_run_dir_resolves_to_its_own_datasets_only(tmp_path):
    cat, run_a, _ = _catalog(tmp_path)
    assert _ids_under([str(run_a)], cat) == ["arm:run_A_bench__examples_32"]


def test_several_runs_accumulate(tmp_path):
    cat, run_a, run_b = _catalog(tmp_path)
    assert sorted(_ids_under([str(run_a), str(run_b)], cat)) == [
        "arm:run_A_bench__examples_32", "arm:run_B_bench__examples_32"]


def test_a_directory_with_no_datasets_returns_nothing(tmp_path):
    cat, _, _ = _catalog(tmp_path)
    empty = tmp_path / "run_C_bench"
    empty.mkdir()
    assert _ids_under([str(empty)], cat) == []


def test_a_dataset_with_no_source_is_skipped(tmp_path):
    cat, run_a, _ = _catalog(tmp_path)
    cat._d.append(_D("arm:sourceless", ""))
    assert _ids_under([str(run_a)], cat) == ["arm:run_A_bench__examples_32"]


def test_the_npz_itself_may_be_named_directly(tmp_path):
    """Pointing at the file, not the directory, still matches it."""
    cat, run_a, _ = _catalog(tmp_path)
    assert _ids_under([str(run_a / "examples_32.npz")], cat) == [
        "arm:run_A_bench__examples_32"]
