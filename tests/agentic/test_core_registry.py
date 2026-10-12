"""Registry: spec grammar, make/lookup, validation, duplicates, lazy entries."""

from __future__ import annotations

import functools
import sys
import textwrap

import pytest

from blockgen.agentic import core
from blockgen.agentic.core import make, parse_spec, register, register_lazy, spec_str

from .fakes import FakeActions, FakeImageGen, FakeRenderer


@pytest.mark.parametrize("spec, expected", [
    ("dsl", ("dsl", None, {})),
    ("openai:gpt-image-1-mini?quality=low&size=1024x1024",
     ("openai", "gpt-image-1-mini", {"quality": "low", "size": "1024x1024"})),
    ("embed?backbone=siglipB&reduce=max", ("embed", None, {"backbone": "siglipB", "reduce": "max"})),
    ("diffusers:stabilityai/sdxl-turbo?steps=4&guidance=0.0&offload=false",
     ("diffusers", "stabilityai/sdxl-turbo", {"steps": 4, "guidance": 0.0, "offload": False})),
    ({"type": "texture_sheet", "tile": 64}, ("texture_sheet", None, {"tile": 64})),
    ({"texture_sheet": {"tile": 64}}, ("texture_sheet", None, {"tile": 64})),
    ({"palette": None}, ("palette", None, {})),
    ({"type": "openai", "arg": "gpt-image-1", "quality": "high"},
     ("openai", "gpt-image-1", {"quality": "high"})),
    ("x?v=null&w=[1,2]", ("x", None, {"v": None, "w": [1, 2]})),
])
def test_parse_spec_forms(spec, expected):
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("spec", [
    "dsl", "openai:gpt-image-1-mini?quality=low&size=1024x1024", "embed?reduce=max&backbone=siglipB",
    {"texture_sheet": {"tile": 64, "cols": 10}}, {"type": "mock", "arg": "render", "split": "train"},
])
def test_spec_str_round_trips(spec):
    s = spec_str(spec)
    assert parse_spec(s) == parse_spec(spec)
    assert spec_str(s) == s                     # canonical: idempotent


def test_spec_str_is_order_independent():
    assert spec_str("e?b=1&a=2") == spec_str("e?a=2&b=1") == "e?a=2&b=1"


def test_bad_specs_raise():
    with pytest.raises(ValueError):
        parse_spec(None)
    with pytest.raises(ValueError):
        parse_spec({"a": {}, "b": {}})
    with pytest.raises(ValueError, match="mapping of params"):
        parse_spec({"examples": 1})


def test_make_function_kind_binds_params():
    fn = make("scorer", "fake_scorer?scale=10")
    assert isinstance(fn, functools.partial) and fn.spec_name == "fake_scorer"
    assert fn.keywords == {"scale": 10}
    with pytest.raises(ValueError, match="takes no ':arg'"):
        make("scorer", "fake_scorer:oops")


def test_make_class_kind_with_arg_and_inject():
    gen = make("image", "fake_image:my-model?usd=0.5")
    assert isinstance(gen, FakeImageGen) and gen.model == "my-model" and gen.usd == 0.5
    # inject reaches only constructors that accept the keyword
    r = make("renderer", "fake_renderer", px=16, unrelated=1)
    assert isinstance(r, FakeRenderer) and r.px == 16
    assert isinstance(make("actions", {"fake_dsl": {}}), FakeActions)


def test_unknown_name_suggests():
    with pytest.raises(KeyError, match="did you mean 'fake_scorer'"):
        core.lookup("scorer", "fake_scorr")
    with pytest.raises(ValueError, match="unknown component kind"):
        core.lookup("nope", "x")


def test_unknown_param_suggests():
    with pytest.raises(TypeError, match=r"unknown parameter\(s\) \['scael'\].*did you mean 'scale'"):
        make("scorer", "fake_scorer?scael=3")


def test_validate_params_var_kw_accepts_anything():
    def f(ep, **kw):
        pass
    core.validate_params(f, {"anything": 1}, where="t")


def test_duplicate_registration_raises(registry_snapshot):
    @register("scorer", "dup_probe")
    def a(ep, att):
        """first"""
    with pytest.raises(ValueError, match="already registered"):
        @register("scorer", "dup_probe")
        def b(ep, att):
            pass
    # re-registering the SAME object (module re-import) is a no-op
    register("scorer", "dup_probe")(a)
    assert core.lookup("scorer", "dup_probe").target is a


def test_aliases_and_describe(registry_snapshot):
    @register("accept", "probe_accept", "probe_alias", needs_dream=False)
    def probe(prev, cand):
        """Accept everything. Second line is not shown."""
        return True, ""
    assert core.lookup("accept", "probe_alias").target is probe
    assert ("probe_accept", "Accept everything. Second line is not shown.") in core.describe("accept")
    assert core.meta("accept", "probe_accept") == {"needs_dream": False}
    assert "probe_alias" not in core.names("accept")


def test_lazy_entry_imported_only_on_make(registry_snapshot, tmp_path, monkeypatch):
    mod = "blockgen_lazy_probe_mod"
    (tmp_path / f"{mod}.py").write_text(textwrap.dedent('''
        class Gen:
            """A lazily imported generator."""
            name = "lazy"
            def __init__(self, model="m", **kw):
                self.model = model
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    register_lazy("image", "lazy_probe", f"{mod}:Gen", "lazy_alias", doc="lazy probe")
    assert mod not in sys.modules
    assert ("lazy_probe", "lazy probe") in core.describe("image")   # describe does not import
    assert mod not in sys.modules
    obj = make("image", "lazy_alias:xyz")
    assert mod in sys.modules and obj.model == "xyz"
    monkeypatch.delitem(sys.modules, mod)


def test_load_builtins_tolerates_missing_suite_modules(monkeypatch):
    monkeypatch.setattr(core, "_BUILTINS_LOADED", False)
    monkeypatch.setattr(core, "_BUILTINS_SKIPPED", {})
    monkeypatch.setattr(core, "BUILTIN_MODULES", ("blockgen.agentic.no_such_module_xyz",))
    with pytest.warns(UserWarning, match="no_such_module_xyz"):
        core.load_builtins()
    assert "blockgen.agentic.no_such_module_xyz" in core._BUILTINS_SKIPPED
    with pytest.raises(KeyError, match="built-in modules not loaded"):
        core.lookup("scorer", "definitely_missing")


def test_load_builtins_raises_on_other_import_errors(monkeypatch, tmp_path):
    (tmp_path / "blockgen_bad_builtin.py").write_text("import a_third_party_pkg_that_is_missing\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(core, "_BUILTINS_LOADED", False)
    monkeypatch.setattr(core, "BUILTIN_MODULES", ("blockgen_bad_builtin",))
    with pytest.raises(ModuleNotFoundError):
        core.load_builtins()
    assert core._BUILTINS_LOADED is False        # retried (and raised) next time, not silent
