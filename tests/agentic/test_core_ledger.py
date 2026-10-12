"""Ledger: caps, propagation, cached calls, summaries, spend gate."""

from __future__ import annotations

import pytest

from blockgen.agentic import core
from blockgen.agentic.core import Budget, BudgetExceeded, Charge, ImageResult, Ledger
from blockgen.agentic.providers import LLMResponse


def llm(usd=0.0, cached=False, role="builder", priced=True):
    return Charge(role=role, kind="llm", model="m", usd=0.0 if cached else usd,
                  would_cost_usd=usd, cached=cached, priced=priced)


def img(usd=0.01, cached=False, n=1):
    return Charge(role="image", kind="image", model="i", usd=0.0 if cached else usd,
                  would_cost_usd=usd, cached=cached, images=n)


def test_usd_cap_checked_before_the_call():
    led = Ledger(Budget(usd=0.10), name="b")
    led.check("llm", 0.05)
    led.charge(llm(0.06))
    led.check("llm", 0.04)                         # exactly at the cap is allowed
    with pytest.raises(BudgetExceeded, match="cap"):
        led.check("llm", 0.05)
    led.check("llm", None)                         # unpriced next call counts as 0


def test_call_and_image_caps_count_misses_only():
    led = Ledger(Budget(max_llm_calls=2, max_images=1))
    led.charge(llm(cached=True))
    led.charge(llm(cached=True))
    led.check("llm")                               # cached calls are free
    led.charge(llm(0.01))
    led.charge(llm(0.01))
    with pytest.raises(BudgetExceeded, match="LLM call cap"):
        led.check("llm")
    led.charge(img(cached=True))
    led.check("image", 0.01)
    led.charge(img())
    with pytest.raises(BudgetExceeded, match="image cap"):
        led.check("image", 0.01)


def test_parent_propagation_and_caps():
    battery = Ledger(Budget(usd=0.05), name="battery")
    arm = battery.child(name="arm")
    b1 = arm.child(Budget(usd=1.0), name="build1")
    b1.charge(llm(0.03))
    assert battery.spent_usd == pytest.approx(0.03) and arm.spent_usd == pytest.approx(0.03)
    b2 = arm.child(Budget(usd=1.0), name="build2")
    with pytest.raises(BudgetExceeded, match="battery"):
        b2.check("llm", 0.03)                      # build cap fine, battery cap not


def test_cached_calls_cost_nothing_but_keep_full_price():
    led = Ledger()
    led.charge(llm(0.02, cached=True))
    assert led.spent_usd == 0.0 and led.would_cost_usd == pytest.approx(0.02)


def test_summary_and_cost_known():
    led = Ledger()
    assert led.cost_known is False                 # nothing charged != free
    led.charge(llm(0.01))
    led.charge(llm(0.01, cached=True, role="judge"))
    led.charge(img(0.005))
    s = led.summary()
    assert s["cost_known"] is True and s["llm_calls"] == 1 and s["images"] == 1
    assert s["usd"] == pytest.approx(0.015) and s["would_cost_usd"] == pytest.approx(0.025)
    assert s["roles"]["judge"]["cached_frac"] == 1.0
    assert s["roles"]["builder"]["billing"] == ["api"]
    led.charge(llm(0.0, priced=False, role="critic"))
    assert led.summary()["cost_known"] is False
    assert led.summary()["roles"]["critic"]["cost_known"] is False


def test_charge_from_llm_and_image():
    resp = LLMResponse(text="x", model="gpt-5-mini", prompt_tokens=1_000_000,
                       completion_tokens=0, cached=True)
    c = Charge.from_llm("critic", resp, billing="api")
    assert c.usd == 0.0 and c.would_cost_usd == pytest.approx(0.25) and c.cached and c.priced
    unpriced = Charge.from_llm("builder", LLMResponse(text="", model="claude-code/sonnet"))
    assert unpriced.priced is False
    res = ImageResult(images=[b"a", b"b"], model="gpt-image-1-mini", cost_usd=0.01)
    ci = Charge.from_image("image", res)
    assert ci.images == 2 and ci.usd == 0.01 and ci.kind == "image"
    assert Charge.from_image("image", ImageResult([b"a"], "m", 0.01, cached=True)).usd == 0.0
    assert set(ci.to_dict()) >= {"role", "kind", "usd", "would_cost_usd", "billing"}


def test_spend_gate(monkeypatch):
    monkeypatch.delenv(core.SPEND_ENV, raising=False)
    assert not core.spend_allowed()
    with pytest.raises(core.SpendNotAllowed, match="--allow-spend"):
        core.require_spend("openai:gpt-5-mini")
    monkeypatch.setenv(core.SPEND_ENV, "1")
    core.require_spend("openai:gpt-5-mini")


def test_billing_of_unwraps_cache_wrappers():
    class Inner:
        billing = "subscription"

    class Wrapper:
        inner = Inner()

    assert core.billing_of(Wrapper()) == "subscription"
    assert core.billing_of(object()) == "api"
