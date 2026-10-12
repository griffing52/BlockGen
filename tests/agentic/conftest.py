"""Shared pytest setup for the agentic suite tests.

Three guarantees for every test in this directory:

* **No spend.** ``BLOCKGEN_ALLOW_SPEND`` is removed from the environment unless the
  test is marked ``@pytest.mark.spend`` (and those are skipped unless the variable
  was set when pytest started). A paying backend therefore hits the spend gate.
* **Fakes registered.** ``fake_*`` components (see fakes.py) are in the registry.
* **Registry isolation on request.** The ``registry_snapshot`` fixture restores the
  global registry after a test that registers throwaway components.
"""

from __future__ import annotations

import copy
import os

import pytest

from blockgen.agentic import core

from .fakes import RecordingTrace, ScriptedLLM, make_fake_kit, register_fakes

_SPEND_AT_START = os.environ.get(core.SPEND_ENV, "") == "1"


def pytest_configure(config: pytest.Config) -> None:
    # pyproject's marker list gains 'spend' in WP7; registering it here keeps
    # --strict-markers runs clean before then.
    config.addinivalue_line("markers", "spend: calls a paid API; skipped unless "
                                       "BLOCKGEN_ALLOW_SPEND=1")
    config.addinivalue_line("markers", "slow: touches the real corpus, a GPU, or the renderer")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if _SPEND_AT_START:
        return
    skip = pytest.mark.skip(reason=f"spends money; set {core.SPEND_ENV}=1 to run")
    for item in items:
        if "spend" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_spend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("spend") is None:
        monkeypatch.delenv(core.SPEND_ENV, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _fakes_registered() -> None:
    register_fakes()


@pytest.fixture
def registry_snapshot():
    """Restore the component registry after the test (register freely inside)."""
    reg = {k: dict(v) for k, v in core._REG.items()}
    ali = {k: dict(v) for k, v in core._ALIASES.items()}
    entries = {k: {n: copy.copy(e) for n, e in v.items()} for k, v in core._REG.items()}
    yield core
    for k in core.KINDS:
        core._REG[k].clear()
        core._REG[k].update({n: entries[k].get(n, e) for n, e in reg[k].items()})
        core._ALIASES[k].clear()
        core._ALIASES[k].update(ali[k])


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def fake_kit():
    return make_fake_kit()


@pytest.fixture
def recording_trace() -> RecordingTrace:
    return RecordingTrace()
