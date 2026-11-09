"""trainer_run: registered, entry resolves, served temperature applies only to the client default."""
from pathlib import Path

from blockgen_server.backends import KINDS, GenerateRequest, ModelSpec, TrainerRunBackend

ROOT = Path(__file__).resolve().parents[3]


def _backend(**extra):
    return TrainerRunBackend(ModelSpec(name="t", kind="trainer_run", extra=extra), ROOT)


def test_registered():
    assert KINDS["trainer_run"] is TrainerRunBackend


def test_served_temperature_replaces_client_default_only():
    b = _backend(temperature=0.7)
    assert b._temperature(GenerateRequest(temperature=1.0)) == 0.7
    assert b._temperature(GenerateRequest(temperature=0.9)) == 0.9
    assert _backend()._temperature(GenerateRequest(temperature=1.0)) == 1.0
