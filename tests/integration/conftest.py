from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lspattern.testing.fingerprints import FingerprintRegistry

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def fingerprint_registry() -> Generator[FingerprintRegistry, None, None]:
    """Load the golden fingerprint registry for integration tests.

    If UPDATE_FINGERPRINTS=1 is set, enables auto-update mode and saves
    changes at the end of the test session.
    """
    path = Path(__file__).parent / "fixtures" / "circuit_fingerprints.json"
    auto_update = os.environ.get("UPDATE_FINGERPRINTS", "0") == "1"

    reg = FingerprintRegistry(path, auto_update=auto_update)
    reg.load()

    if auto_update:
        print("\n[UPDATE_FINGERPRINTS=1] Auto-update mode enabled for fingerprints")

    yield reg

    # Save if any fingerprints were updated during the session
    if auto_update and reg.save_if_updated():
        print(f"\n[UPDATE_FINGERPRINTS] Saved updated fingerprints to {path}")
