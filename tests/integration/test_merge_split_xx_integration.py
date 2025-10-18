from __future__ import annotations

from pathlib import Path

import pytest

from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def _registry() -> FingerprintRegistry:
    path = Path(__file__).parent / "fixtures" / "circuit_fingerprints.json"
    reg = FingerprintRegistry(path)
    reg.load()
    return reg


def test_merge_split_xx_compile_and_metadata() -> None:
    from examples.merge_split_xx_error_sim import create_circuit

    d = 3
    circuit = create_circuit(d=d, noise=0.0)

    # Basic metadata checks
    assert circuit.num_qubits > 0
    assert circuit.num_observables >= 1

    # Fingerprint regression
    reg = _registry()
    fp = CircuitFingerprint.from_circuit(f"merge_split_xx_d{d}", circuit)
    ok, err = reg.verify(fp)
    assert ok, err


@pytest.mark.slow
def test_merge_split_xx_dem() -> None:
    from examples.merge_split_xx_error_sim import create_circuit
    import pymatching as pm

    d = 3
    noise = 1e-2
    circuit = create_circuit(d=d, noise=noise)

    dem = circuit.detector_error_model(decompose_errors=True)
    _ = pm.Matching.from_detector_error_model(dem)
