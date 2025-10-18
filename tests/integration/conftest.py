from __future__ import annotations

from pathlib import Path

import pytest

from lspattern.testing.fingerprints import FingerprintRegistry


@pytest.fixture(scope="session")
def fingerprint_registry() -> FingerprintRegistry:
    """Load the golden fingerprint registry for integration tests."""
    path = Path(__file__).parent / "fixtures" / "circuit_fingerprints.json"
    reg = FingerprintRegistry(path)
    reg.load()
    return reg
