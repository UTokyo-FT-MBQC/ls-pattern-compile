from __future__ import annotations

import pymatching as pm
import pytest

from examples.memory_error_sim import create_circuit
from lspattern.consts import InitializationState
from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def test_initialization_plus_compile_and_metadata(fingerprint_registry: FingerprintRegistry) -> None:
    d = 3
    circuit = create_circuit(d=d, noise=0.0, init_type=InitializationState.PLUS)

    # Basic metadata checks
    assert circuit.num_qubits > 0
    assert circuit.num_observables >= 1

    # Fingerprint regression
    fp = CircuitFingerprint.from_circuit(f"initialization_plus_d{d}", circuit)
    ok, err = fingerprint_registry.verify(fp)
    assert ok, err


@pytest.mark.slow
def test_initialization_plus_dem() -> None:
    d = 3
    noise = 1e-2
    circuit = create_circuit(d=d, noise=noise, init_type=InitializationState.PLUS)

    dem = circuit.detector_error_model(decompose_errors=True)
    _ = pm.Matching.from_detector_error_model(dem)
