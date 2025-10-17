from __future__ import annotations

from pathlib import Path

import pytest

from lspattern.consts import InitializationState
from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def _registry() -> FingerprintRegistry:
    path = Path(__file__).parent / "fixtures" / "circuit_fingerprints.json"
    reg = FingerprintRegistry(path)
    reg.load()
    return reg


def test_memory_zero_compile_and_metadata() -> None:
    from examples.memory_error_sim import create_circuit

    d = 3
    circuit = create_circuit(d=d, noise=0.0, init_type=InitializationState.ZERO)

    # Basic metadata checks
    assert circuit.num_qubits > 0
    assert circuit.num_observables >= 1

    # Fingerprint regression
    reg = _registry()
    fp = CircuitFingerprint.from_circuit(f"memory_zero_d{d}", circuit)
    ok, err = reg.verify(fp)
    assert ok, err


@pytest.mark.slow
def test_memory_zero_quick_error_sim() -> None:
    from examples.memory_error_sim import create_circuit
    import pymatching as pm

    d = 3
    noise = 1e-2
    circuit = create_circuit(d=d, noise=noise, init_type=InitializationState.ZERO)

    dem = circuit.detector_error_model(decompose_errors=True)
    _ = pm.Matching.from_detector_error_model(dem)
