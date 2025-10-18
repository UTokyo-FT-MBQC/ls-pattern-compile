from __future__ import annotations

import pytest

from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry


def test_merge_split_compile_and_metadata(fingerprint_registry: FingerprintRegistry) -> None:
    from examples.merge_split_error_sim import create_circuit

    d = 3
    circuit = create_circuit(d=d, noise=0.0)

    # Basic metadata checks
    assert circuit.num_qubits > 0
    assert circuit.num_observables >= 1

    # Fingerprint regression
    fp = CircuitFingerprint.from_circuit(f"merge_split_d{d}", circuit)
    ok, err = fingerprint_registry.verify(fp)
    assert ok, err


@pytest.mark.slow
def test_merge_split_dem() -> None:
    from examples.merge_split_error_sim import create_circuit
    import pymatching as pm

    d = 3
    noise = 1e-2
    circuit = create_circuit(d=d, noise=noise)

    # Ensure a detector error model and matching graph can be built
    dem = circuit.detector_error_model(decompose_errors=True)
    _ = pm.Matching.from_detector_error_model(dem)
