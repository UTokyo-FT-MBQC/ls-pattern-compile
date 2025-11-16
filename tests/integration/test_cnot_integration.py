from __future__ import annotations

import pymatching as pm
import pytest

from examples.cnot_error_sim import create_circuit
from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def test_cnot_compile_and_metadata(fingerprint_registry: FingerprintRegistry) -> None:
    """Test CNOT circuit compilation and verify circuit fingerprint."""
    d = 3
    circuit = create_circuit(d=d, noise=0.0)

    # Basic metadata checks
    assert circuit.num_qubits > 0
    assert circuit.num_observables >= 1

    # Fingerprint regression
    fp = CircuitFingerprint.from_circuit(f"cnot_d{d}", circuit)
    ok, err = fingerprint_registry.verify(fp)
    assert ok, err


@pytest.mark.slow
def test_cnot_dem() -> None:
    """Test CNOT detector error model construction and matching graph generation."""
    d = 3
    noise = 1e-2
    circuit = create_circuit(d=d, noise=noise)

    # Ensure a detector error model and matching graph can be built
    dem = circuit.detector_error_model(decompose_errors=True)
    _ = pm.Matching.from_detector_error_model(dem)
