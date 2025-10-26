from __future__ import annotations

import json
from pathlib import Path

import stim

from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def _toy_circuit() -> stim.Circuit:
    c = stim.Circuit()
    c.append_operation("H", [0])
    c.append_operation("CNOT", [0, 1])
    c.append_operation("M", [0, 1])
    c.append_operation("TICK")
    # No observables needed for this unit test
    return c


def test_fingerprint_roundtrip(tmp_path: Path) -> None:
    circ = _toy_circuit()
    fp = CircuitFingerprint.from_circuit("toy", circ)
    assert fp.sha256
    assert fp.num_qubits >= 2
    assert fp.num_detectors >= 0
    assert fp.num_observables >= 0

    reg_path = tmp_path / "golden.json"
    reg = FingerprintRegistry(reg_path)
    reg.set(fp)
    reg.save()

    # load and verify
    reg2 = FingerprintRegistry(reg_path)
    reg2.load()
    ok, err = reg2.verify(fp)
    assert ok and err is None

    # sanity check on file content
    data = json.loads(reg_path.read_text())
    assert "toy" in data and "sha256" in data["toy"]
